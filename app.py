import streamlit as st
import cv2
import mediapipe as mp
import time
import tensorflow as tf
import numpy as np
import math
from PIL import Image

# Set page config
st.set_page_config(page_title="Sign Language Recognition", layout="wide")

# Title
st.title("Sign Language Recognition")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("signlanguage3.h5")
    return model

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Initialize MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Class labels
class_labels = ['AAROHAN', 'Am', 'Are', 'Fine', 'Hello', 'How', 'I', 'To', 'Welcome', 'You']

# Display current prediction
prediction_placeholder = st.empty()
confidence_placeholder = st.empty()

# Image display columns
col1, col2 = st.columns(2)
with col1:
    main_image_placeholder = st.empty()
with col2:
    hand_image_placeholder = st.empty()

def preprocess_for_model(img):
    """Preprocess image to match the model's expected input."""
    if img is None or img.size == 0:
        return None

    # Resize to 50x50 to match training dimensions
    img = cv2.resize(img, (50, 50))
    
    # Convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Normalize to range [0, 1]
    img = img.astype("float32") / 255.0
    
    # Add channel dimension (grayscale = 1 channel)
    img = np.expand_dims(img, axis=-1)
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def crop_hands(img, handLms):
    """Crop the hand region from the image using hand landmarks."""
    h, w, c = img.shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0

    for lm in handLms.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min = min(x, x_min)
        y_min = min(y, y_min)
        x_max = max(x, x_max)
        y_max = max(y, y_max)

    # Add padding
    padding = 40
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)

    # Crop the image
    cropped_img = img[y_min:y_max, x_min:x_max]

    return cropped_img, (x_min, y_min, x_max, y_max)

def show_pred(img, index, x_min, y_min, confidence=None):
    if 0 <= index < len(class_labels):
        confidence_text = f" ({confidence:.2f})" if confidence is not None else ""
        cv2.putText(
            img,
            f"Sign: {class_labels[index]}{confidence_text}",
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
    return img

# Start the webcam
start_button = st.button("Start Camera")

if start_button:
    st.warning("Press Q in the video window or click 'Stop' to quit")
    stop_button = st.button("Stop")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    imgSize = 224  # For hand cropping/display
    
    while cap.isOpened() and not stop_button:
        success, img = cap.read()
        if not success:
            st.error("Failed to capture image from camera")
            break
            
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRgb)

        imgLandmarksOnly = np.ones((720, 1280, 3), np.uint8) * 255
        
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # Get cropped hand image with landmarks only (white background)
                cropped_image, bbox = crop_hands(imgLandmarksOnly, handLms)
                
                if cropped_image.size == 0:
                    continue

                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Draw landmarks on original image and white background
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                mpDraw.draw_landmarks(imgLandmarksOnly, handLms, mpHands.HAND_CONNECTIONS)

                h, w = cropped_image.shape[:2]
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                aspect_ratio = h / w

                # Center the hand in a square white image
                if aspect_ratio > 1:
                    k = imgSize / h
                    wCal = int(math.ceil(k * w))
                    imgResize = cv2.resize(cropped_image, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) // 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = int(math.ceil(k * h))
                    imgResize = cv2.resize(cropped_image, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) // 2)
                    imgWhite[hGap:hGap + hCal, :] = imgResize
                
                # Preprocess for model prediction
                processed_img = preprocess_for_model(imgWhite)
                
                if processed_img is not None:
                    # Make prediction
                    prediction = model.predict(processed_img, verbose=0)
                    confidence = np.max(prediction)
                    index = np.argmax(prediction)
                    
                    # Only show high confidence predictions
                    if confidence > 0.7:
                        img = show_pred(img, index, x_min, y_min, confidence)
                        prediction_placeholder.subheader(f"Prediction: {class_labels[index]}")
                        confidence_placeholder.info(f"Confidence: {confidence:.2f}")
                
                # Display the cropped hand image
                hand_image_placeholder.image(imgWhite, caption="Processed Hand", channels="RGB")
        
        # Calculate and display FPS
        cTime = time.time()
        fps = int(1 / (cTime - time.time() + 0.001))
        cv2.putText(img, f"FPS:{fps}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        
        # Display the main camera feed
        main_image_placeholder.image(img, caption="Camera Feed", channels="BGR")
        
        # Check for key press (for compatibility with original code)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Release camera resources
    cap.release()
    cv2.destroyAllWindows()