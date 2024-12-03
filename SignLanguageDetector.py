import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from collections import deque
import time
import os
import cv2
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

from ASLDataset import ASLDataset
from ASLNet import ASLNet


class SignLanguageDetector:
    def __init__(self, device=None, model_path="asl_model.pth"):
        self.model = None
        self.model_path = model_path
        self.gesture_history = deque(maxlen=30)
        self.classes = []  # will be updated based on dataset
        self.last_prediction_time = time.time()
        self.prediction_cooldown = 0.5
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: ", self.device)

        # Set up transforms
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Training transforms with augmentation
        self.train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((200, 200)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def _create_model(self):
        """Creates a new ASLNet model"""
        model = ASLNet(num_classes=len(self.classes))
        model = model.to(self.device)
        return model

    def train(self, train_dir, epochs=50, batch_size=32, validation_split=0.2, samples_per_class=3000):
        """Trains the model on the ASL dataset"""

        # First, get the list of classes
        self.classes = sorted([d for d in os.listdir(train_dir)
                               if os.path.isdir(os.path.join(train_dir, d))])
        print(f"Found classes: {self.classes}")

        print("Loading and preprocessing data...")

        # Create train and validation datasets
        train_dataset = ASLDataset(
            train_dir,
            self.classes,
            transform=self.train_transform,
            samples_per_class=samples_per_class,
            split='train',
            validation_split=validation_split
        )

        val_dataset = ASLDataset(
            train_dir,
            self.classes,
            transform=self.transform,
            samples_per_class=samples_per_class,
            split='val',
            validation_split=validation_split
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # Create model
        self.model = self._create_model()

        # Loss, optimizer, scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Mix precision
        scaler = torch.amp.GradScaler('cuda')

        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        patience = 5

        epoch_pbar = tqdm(range(epochs), desc='Training Progress', unit='epoch', leave=False, dynamic_ncols=True,
                          position=0)

        # Training loop
        for epoch in epoch_pbar:
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            curr_lr = float(optimizer.param_groups[0]["lr"])

            batch_pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs} [Train]', leave=False,
                              unit='batch', dynamic_ncols=True, position=0)
            start_time = time.time()

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                # Mixed precision training
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                # Update training metrics in progress bar
                current_loss = train_loss / (batch_pbar.n + 1)
                current_acc = 100 * train_correct / train_total
                elapsed_time = time.time() - start_time
                batch_pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{current_acc:.2f}%',
                    'time': f'{elapsed_time:.1f}s'
                })
                batch_pbar.update()

            train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total

            batch_pbar.close()

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]', leave=False, unit='batch',
                            dynamic_ncols=True, position=0)

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                    # Update validation metrics
                    current_val_loss = val_loss / (val_pbar.n + 1)
                    current_val_acc = 100 * val_correct / val_total
                    val_pbar.set_postfix({
                        'loss': f'{current_val_loss:.4f}',
                        'acc': f'{current_val_acc:.2f}%'
                    })
                    val_pbar.update()

            val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            val_pbar.close()

            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'train_acc': f'{train_acc:.2f}%',
                'val_loss': f'{val_loss:.4f}',
                'val_acc': f'{val_acc:.2f}%',
                'lr': f'{curr_lr:.6f}'
            })
            epoch_pbar.update()

            # Print progress
            print(f'Epoch [{epoch + 1}/{epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                  f'lr: {curr_lr:.6f}'
                  )

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)

    def test(self, test_dir):
        """Evaluates model on test dataset"""
        self.model.eval()
        test_dataset = ASLDataset(
            test_dir,
            self.classes,
            transform=self.transform,
            split='test'
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        correct = 0
        total = 0
        predictions = []
        true_labels = []

        # Create progress bar for testing
        test_pbar = tqdm(test_loader, desc='Testing')

        with torch.no_grad():
            for images, labels in test_pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

                # Update progress bar
                test_pbar.set_postfix({
                    'acc': f'{100.0 * correct / total:.2f}%'
                })
                test_pbar.update()

        test_pbar.close()

        accuracy = 100 * correct / total

        # Get unique classes present in the test set
        unique_classes = sorted(set(true_labels))
        class_names = [self.classes[i] for i in unique_classes]

        # Calculate metrics only for classes present in test set
        true_labels_array = np.array(true_labels)
        predictions_array = np.array(predictions)

        precision = []
        recall = []
        f1 = []

        for class_idx in unique_classes:
            class_mask = true_labels_array == class_idx
            true_class = true_labels_array[class_mask]
            pred_class = predictions_array[class_mask]

            if len(true_class) > 0:
                precision.append(precision_score(true_class == class_idx, pred_class == class_idx, zero_division=0))
                recall.append(recall_score(true_class == class_idx, pred_class == class_idx, zero_division=0))
                f1.append(f1_score(true_class == class_idx, pred_class == class_idx, zero_division=0))

        return {
            'accuracy': accuracy,
            'precision': np.array(precision),
            'recall': np.array(recall),
            'f1': np.array(f1),
            'class_mapping': dict(zip(range(len(class_names)), class_names))
        }

    def save_model(self, filepath):
        """Saves the trained model"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'classes': self.classes
            }, filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save. Please train the model first.")

    def load_model(self, filepath):
        """Loads a previously trained model"""
        if not os.path.exists(filepath):
            print(f"Model file not found: {filepath}")
            return

        checkpoint = torch.load(filepath, map_location=self.device)
        self.model = self._create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.classes = checkpoint['classes']
        self.model.eval()
        print(f"Model loaded from {filepath}")

    def preprocess_landmarks(self, landmarks):
        """Converts MediaPipe landmarks to an image representation"""
        image = np.zeros((200, 200), dtype=np.float32)

        # Draw landmarks
        for i in range(0, len(landmarks), 2):
            x = int(landmarks[i] * 200)
            y = int(landmarks[i + 1] * 200)
            if 0 <= x < 200 and 0 <= y < 200:
                cv2.circle(image, (x, y), 3, 1.0, -1)

        # Draw connections
        connections = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                       (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
                       (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]

        for start_idx, end_idx in connections:
            start_x = int(landmarks[start_idx * 2] * 200)
            start_y = int(landmarks[start_idx * 2 + 1] * 200)
            end_x = int(landmarks[end_idx * 2] * 200)
            end_y = int(landmarks[end_idx * 2 + 1] * 200)

            if (0 <= start_x < 200 and 0 <= start_y < 200 and
                    0 <= end_x < 200 and 0 <= end_y < 200):
                cv2.line(image, (start_x, start_y), (end_x, end_y), 0.5, 1)

        return image

    def predict_letter(self, landmarks):
        """Predicts the letter from MediaPipe landmarks"""
        current_time = time.time()
        if current_time - self.last_prediction_time < self.prediction_cooldown:
            return None

        # Load the model
        if self.model is None:
            self.load_model(self.model_path)

        # Convert landmarks to list
        landmark_list = []
        for landmark in landmarks:
            landmark_list.extend([landmark.x, landmark.y])

        # Preprocess landmarks
        image = self.preprocess_landmarks(landmark_list)
        image = Image.fromarray((image * 255).astype(np.uint8))
        image = self.transform(image).unsqueeze(0).to(self.device)

        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = self.classes[predicted.item()]

        # Update history and timing
        self.gesture_history.append(predicted_class)
        self.last_prediction_time = current_time

        return self._get_stable_prediction()

    def _get_stable_prediction(self):
        """Returns the most common prediction in recent history"""
        if len(self.gesture_history) < 10:
            return None

        from collections import Counter
        counter = Counter(self.gesture_history)
        most_common = counter.most_common(1)[0]

        if most_common[1] >= len(self.gesture_history) * 0.7:
            return most_common[0]
        return None
