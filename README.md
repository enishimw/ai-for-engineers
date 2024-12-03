# ASL Hand Gesture Recognition System

This project implements a real-time American Sign Language (ASL) recognition system using computer vision and deep learning. The system can detect hand gestures through a webcam, recognize ASL letters, and provide mouse control functionality.

## Features

- Real-time ASL letter recognition using webcam input
- Mouse control through hand gestures
- Click detection based on finger pinch gestures
- Support for both left and right hand detection
- Training pipeline for custom ASL datasets
- Model evaluation and metrics calculation

## Requirements

- Python 3.7+
- PyTorch
- OpenCV (cv2)
- MediaPipe
- PyAutoGUI
- NumPy
- Pillow (PIL)
- tqdm
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd asl-recognition
```

2. Install the required packages:
```bash
pip install torch torchvision opencv-python mediapipe pyautogui numpy pillow tqdm scikit-learn
```

## Usage

### Running the Application

1. Start the real-time recognition system:
```bash
python main.py
```

This will open your webcam and start detecting:
- Left hand: ASL letter recognition
- Right hand: Mouse control and click detection

### Training the Model

1. Download the ASL Alphabet dataset from Kaggle
2. Make sure it follows the following structure
```bash
dataset/
├── asl_alphabet_train/
│   ├── A/
│   ├── B/
│   └── ...
└── asl_alphabet_test/
    ├── A_test.jpg
    ├── B_test.jpg
    └── ...
```

3. Run the training script:
```bash
python train.py
```

The script will:
- Train the model on the ASL Alphabet dataset
- Evaluate performance on the test set
- Save the trained model as `asl_model.pth`

## Model Architecture

The system uses a custom CNN architecture implemented in PyTorch, with the following features:
- Grayscale image input (200x200)
- Data augmentation during training
- Early stopping and learning rate scheduling
- Mixed precision training
- Model evaluation with precision, recall, and F1 score metrics

## Gestures

### Mouse Control (Right Hand)
- Move cursor: Pinch gesture
- Click: Quick pinch-release-pinch sequence

### ASL Recognition (Left Hand)
- Shows detected ASL letter with configurable cooldown
- Supports the full ASL alphabet

## Performance Metrics

The model is evaluated on:
- Overall accuracy
- Per-class precision
- Per-class recall
- Per-class F1 score

Results are displayed after training and testing phases.

## Team

- Elvis Nishimwe Ndabaye (enishimw@andrew.cmu.edu)
- Angelique Uwamahoro (auwamaho@andrew.cmu.edu)
- Steven Nahimana (snahiman@andrew.cmu.edu)

## Acknowledgments

- MediaPipe for hand landmark detection
- PyTorch for the deep learning framework
- Kaggle for the dataset
