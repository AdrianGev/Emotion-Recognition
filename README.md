# Emotion Recognition

A real-time emotion detection application using computer vision techniques to identify facial expressions.

## Features

- Real-time face detection
- Emotion classification (Happy, Sad, Angry, Surprise, Neutral)
- Live video feed with emotion predictions
- Confidence scores for detected emotions
- Clean and intuitive user interface

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Tkinter
- Pillow (PIL)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AdrianGev/Emotion-Recognition.git
   cd Emotion-Recognition
   ```

2. Install required packages:
   ```bash
   pip install opencv-python numpy pillow
   ```

## Usage

Run the application:
```bash
python emotion_detector.py
```

The application will open a window showing the video feed from your webcam. It will automatically detect faces and display the detected emotion with a confidence score.

## How it Works

The application uses:
- Haar Cascade Classifiers for face, eye, and smile detection
- Region-based feature extraction for emotion analysis
- Rule-based classification system for emotion prediction
- Real-time video processing and display

## License

MIT License

## Author

AdrianGev
