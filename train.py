import os
import torch
from SignLanguageDetector import SignLanguageDetector

# Initialize detector
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize detector
detector = SignLanguageDetector()
root_dir = "/kaggle/input/asl-alphabet"

# Train model
train_dir = os.path.join(root_dir, "asl_alphabet_train/asl_alphabet_train")
detector.train(train_dir, epochs=2, batch_size=30)


# Test the model
test_dir = os.path.join(root_dir, "asl_alphabet_test/asl_alphabet_test")
test_metrics = detector.test(test_dir)

# Print results
print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
print("\nPer-class metrics:")
for class_idx, class_name in test_metrics['class_mapping'].items():
    print(f"{class_name}:")
    print(f"  Precision: {test_metrics['precision'][class_idx]:.3f}")
    print(f"  Recall: {test_metrics['recall'][class_idx]:.3f}")
    print(f"  F1: {test_metrics['f1'][class_idx]:.3f}")

# Save the trained model
detector.save_model("asl_model.pth")


# detector.load_model("asl_model.pth")