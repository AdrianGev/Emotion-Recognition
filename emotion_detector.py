import cv2
import numpy as np
import sys
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class EmotionDetector:
    def __init__(self):
        self.emotions = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        # Load haar cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces, gray

    def analyze_face(self, gray, x, y, w, h):
        face_roi = gray[y:y+h, x:x+w]
        
        # Detect smiles and eyes
        smiles = self.smile_cascade.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=20)
        eyes = self.eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5)
        
        # Calculate regions of interest
        h_third = h // 3
        w_third = w // 3
        
        # Extract regions
        forehead = face_roi[0:h_third, :]
        eyes_region = face_roi[h_third:2*h_third, :]
        mouth_region = face_roi[2*h_third:h, :]
        
        # Calculate features
        features = {
            'smile_count': len(smiles),
            'eye_count': len(eyes),
            'forehead_intensity': np.mean(forehead),
            'eyes_intensity': np.mean(eyes_region),
            'mouth_intensity': np.mean(mouth_region),
            'mouth_variance': np.var(mouth_region),
            'edge_intensity': np.mean(cv2.Canny(face_roi, 100, 200))
        }
        
        return features

    def predict_emotion(self, frame):
        try:
            faces, gray = self.detect_faces(frame)
            
            if len(faces) == 0:
                return "No face", 0.0
            
            # Process the largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            features = self.analyze_face(gray, x, y, w, h)
            
            # Enhanced emotion classification based on features
            if features['smile_count'] > 0 and features['mouth_variance'] > 1000:
                return "Happy", 0.85
            elif features['edge_intensity'] > 50 and features['forehead_intensity'] < 100:
                return "Angry", 0.75
            elif features['eye_count'] < 2 and features['mouth_intensity'] < 100:
                return "Sad", 0.7
            elif features['edge_intensity'] > 40 and features['forehead_intensity'] > 120:
                return "Surprise", 0.8
            else:
                return "Neutral", 0.9
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "Error", 0.0

class EmotionDetectorApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Emotion Detector")
        
        # Set minimum window size and initial size
        self.window.minsize(800, 800)
        self.window.geometry("1024x900")
        
        # Make the window resizable
        self.window.resizable(True, True)
        
        # Configure grid weights
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)

        # Create main frame with padding
        self.main_frame = ttk.Frame(window, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure main frame grid weights
        self.main_frame.grid_rowconfigure(0, weight=3)  # Video feed takes more space
        self.main_frame.grid_rowconfigure(1, weight=0)  # Status row
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Create video frame with fixed minimum size
        self.video_frame = ttk.Frame(self.main_frame, width=640, height=480)
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.video_frame.grid_propagate(False)  # Prevent frame from shrinking
        self.video_frame.grid_columnconfigure(0, weight=1)
        self.video_frame.grid_rowconfigure(0, weight=1)

        # Create video display label
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # Create status frame
        self.status_frame = ttk.Frame(self.main_frame)
        self.status_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.status_frame.grid_columnconfigure(0, weight=1)

        # Create status label
        self.status_label = ttk.Label(
            self.status_frame,
            text="Initializing emotion detection...",
            font=('Arial', 12),
            wraplength=700
        )
        self.status_label.grid(row=0, column=0, pady=5)

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Could not open camera. Please check permissions.")
            sys.exit(1)

        # Set camera resolution to a more standard size
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Initialize emotion detector
        self.emotion_detector = EmotionDetector()
        self.status_label.config(text="Ready! Detecting emotions...")
        
        # Start video update
        self.update_frame()

    def resize_frame(self, frame, target_width, target_height):
        """Safely resize frame maintaining aspect ratio"""
        if target_width <= 0 or target_height <= 0:
            return frame
            
        height, width = frame.shape[:2]
        
        # Calculate scaling factor to fit in window
        scale = min(target_width/width, target_height/height)
        
        if scale <= 0:
            return frame
            
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Ensure dimensions are valid
        if new_width <= 0 or new_height <= 0:
            return frame
            
        return cv2.resize(frame, (new_width, new_height))

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Get the current size of the video frame
            frame_width = self.video_frame.winfo_width()
            frame_height = self.video_frame.winfo_height()
            
            # Only resize if we have valid dimensions
            if frame_width > 100 and frame_height > 100:  # Minimum reasonable size
                frame = self.resize_frame(frame, frame_width, frame_height)
            
            # Detect faces and predict emotion
            faces, gray = self.emotion_detector.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Predict emotion for this face
                emotion, confidence = self.emotion_detector.predict_emotion(frame)
                
                # Display emotion text with confidence
                text = f"{emotion} ({confidence:.2f})"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                # Update status label with current emotion
                self.status_label.config(text=f"Current Emotion: {emotion} (Confidence: {confidence:.2f})")

            # Convert frame to tkinter format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=image)
            
            # Update label
            self.video_label.configure(image=photo)
            self.video_label.image = photo

        # Schedule next update
        self.window.after(30, self.update_frame)

    def cleanup(self):
        if self.cap.isOpened():
            self.cap.release()
        self.window.destroy()

def main():
    # Create Info.plist if it doesn't exist
    if not os.path.exists("Info.plist"):
        plist_content = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>NSCameraUsageDescription</key>
    <string>This app needs access to camera for emotion detection.</string>
</dict>
</plist>"""
        with open("Info.plist", "w") as f:
            f.write(plist_content)

    # Create tkinter window
    root = tk.Tk()
    app = EmotionDetectorApp(root)
    
    # Set up cleanup on window close
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()
