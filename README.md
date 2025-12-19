# 👋 Sign Language Interpreter - Real-Time Gesture Recognition

A deep learning-powered sign language interpreter that provides real-time gesture recognition using CNN architecture and MediaPipe hand tracking. This system achieves 99.6% validation accuracy and can recognize 10 different sign language gestures at 30+ FPS.

## 🎯 Overview

This project implements a real-time sign language recognition system that combines:
- **Convolutional Neural Networks (CNN)** for accurate gesture classification
- **MediaPipe Hand Tracking** for robust hand landmark detection (21 landmarks)
- **Streamlit Web Interface** for easy-to-use deployment
- **High Accuracy** with 99.6% validation accuracy on the test dataset

The system captures hand gestures through a webcam, processes them using MediaPipe for hand detection and landmark extraction, and classifies them using a trained CNN model to recognize sign language gestures in real-time.

## ✨ Features

- **Real-Time Recognition**: Processes gestures at 30+ FPS for smooth real-time interaction
- **MediaPipe Integration**: Leverages MediaPipe's 21-landmark hand tracking for robust detection
- **High Accuracy**: Achieves 99.6% validation accuracy with confidence filtering
- **Confidence Filtering**: Only displays predictions with >70% confidence for reliability
- **Web-Based Interface**: Built with Streamlit for easy access and deployment
- **Preprocessing Pipeline**: Automated hand ROI extraction and normalization
- **Multi-Gesture Support**: Recognizes 10 different sign language gestures
- **Visual Feedback**: Shows hand landmarks, bounding boxes, and confidence scores

## 🏗️ Project Structure

```
Sign-language-interpreter-improved-/
├── app.py                      # Streamlit web application for real-time recognition
├── trainmodel.ipynb            # Model training notebook with architecture definition
├── createdataset.ipynb         # Dataset collection tool using webcam
├── notebook.ipynb              # Additional experiments and testing
├── test.ipynb                  # Model testing and evaluation
├── signlanguage3.h5            # Pre-trained model (70MB)
├── dataset2/                   # Training dataset directory
│   ├── AAROHAN/               # Gesture class samples
│   ├── Am/
│   ├── Are/
│   ├── Fine/
│   ├── Hello/
│   ├── How/
│   ├── I/
│   ├── To/
│   ├── Welcome/
│   └── You/
└── README.md                   # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam for real-time gesture recognition
- GPU (optional, but recommended for training)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Arka077/Sign-language-interpreter-improved-.git
   cd Sign-language-interpreter-improved-
   ```

2. **Install required dependencies**
   ```bash
   pip install streamlit opencv-python mediapipe tensorflow numpy pillow
   ```

3. **Verify the model file exists**
   ```bash
   ls signlanguage3.h5
   ```
   The pre-trained model should be approximately 70MB.

## 🎮 Usage

### Running the Application

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**
   - The app will automatically open in your default browser
   - Or navigate to `http://localhost:8501`

3. **Use the gesture recognition**
   - Click the "Start Camera" button
   - Position your hand in front of the webcam
   - The system will detect your hand and display:
     - Main camera feed with hand landmarks
     - Processed hand image (cropped and normalized)
     - Predicted gesture with confidence score
   - Only predictions with >70% confidence are displayed
   - Press 'Q' in the video window or click "Stop" to quit

### Webcam-Based Recognition

The system performs the following steps in real-time:
1. Captures video frames from your webcam
2. Detects hands using MediaPipe
3. Extracts 21 hand landmarks
4. Crops and normalizes the hand region
5. Classifies the gesture using the CNN model
6. Displays the prediction with confidence score

## 🧠 Model Architecture

The CNN model consists of:

### Convolutional Blocks
```
Input: 50x50x1 (grayscale image)

Conv Block 1:
├── Conv2D(256 filters, 3x3 kernel, ReLU)
├── BatchNormalization
├── MaxPooling2D(2x2)
└── Dropout(0.5)

Conv Block 2:
├── Conv2D(256 filters, 3x3 kernel, ReLU)
├── BatchNormalization
├── MaxPooling2D(2x2)
└── Dropout(0.5)

Conv Block 3:
├── Conv2D(512 filters, 3x3 kernel, ReLU)
├── BatchNormalization
├── MaxPooling2D(2x2)
└── Dropout(0.5)

Flatten
```

### Fully Connected Layers
```
├── Dense(512, ReLU)
├── Dropout(0.5)
├── Dense(256, ReLU)
├── Dropout(0.5)
├── Dense(64, ReLU)
├── Dropout(0.5)
└── Dense(10, Softmax)  # 10 gesture classes
```

**Total Parameters**: ~millions (optimized for accuracy and speed)

## 📊 Training Details

### Dataset
- **Total Images**: 5,003
  - Training: 4,003 images
  - Validation: 1,000 images
- **Image Size**: 50x50 pixels
- **Color Mode**: Grayscale
- **Classes**: 10 gesture types

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 128
- **Early Stopping**: 
  - Patience: 20 epochs
  - Monitor: validation loss
  - Min delta: 0.00001
- **Best Epoch**: 36 (out of 56 total)

### Performance Metrics
- **Validation Accuracy**: 99.6% (epoch 36)
- **Training Accuracy**: 99.6%
- **Best Validation Loss**: 3.5542e-07
- **Training stopped at**: Epoch 56 (restored best weights from epoch 36)

## 🔍 Preprocessing Pipeline

The preprocessing pipeline ensures consistent input to the model:

1. **Hand Detection**: MediaPipe detects hands and extracts 21 landmarks
2. **Bounding Box Calculation**: Creates a box around detected landmarks with 40px padding
3. **ROI Extraction**: Crops the hand region from the frame
4. **Aspect Ratio Preservation**: Centers the hand in a 224x224 white canvas
5. **Model Input Preparation**:
   - Resize to 50x50 pixels
   - Convert to grayscale
   - Normalize pixel values to [0, 1]
   - Add channel dimension (50x50x1)
   - Add batch dimension (1x50x50x1)

## 📸 Supported Gestures

The system recognizes the following 10 sign language gestures:

| Gesture Class | Description | Dataset Size (approx.) |
|--------------|-------------|----------------------|
| AAROHAN | Event/Festival name | ~500 images |
| I | Personal pronoun | ~500 images |
| Am | Verb (to be) | ~500 images |
| Are | Verb (to be) | ~500 images |
| Fine | Adjective/Response | ~500 images |
| Hello | Greeting | ~500 images |
| How | Question word | ~500 images |
| To | Preposition | ~500 images |
| Welcome | Greeting/Response | ~500 images |
| You | Personal pronoun | ~500 images |

These gestures combine to form basic phrases like "Hello, how are you?" and "I am fine, welcome to AAROHAN."

## 🛠️ Creating Custom Dataset

To create your own dataset or add new gestures:

1. **Open the dataset creation notebook**
   ```bash
   jupyter notebook createdataset.ipynb
   ```

2. **Configure the gesture class**
   ```python
   folder = "YourGestureName"  # Change this to your gesture name
   ```

3. **Run the webcam capture**
   - Position your hand in front of the camera
   - Press 'S' to save a frame
   - Collect at least 300-500 images per gesture
   - Press 'Q' to quit

4. **Organize the dataset**
   ```
   dataset2/
   └── YourGestureName/
       ├── Image1.jpg
       ├── Image2.jpg
       └── ...
   ```

5. **Tips for good dataset quality**:
   - Use various hand positions and angles
   - Include different lighting conditions
   - Use different backgrounds
   - Maintain consistent gesture formation
   - Ensure hands are clearly visible

## 🏋️ Training the Model

To train the model with your custom dataset:

1. **Prepare your dataset** in the `dataset2/` directory

2. **Open the training notebook**
   ```bash
   jupyter notebook trainmodel.ipynb
   ```

3. **Configure training parameters** (optional)
   ```python
   batch_size = 128
   image_size = (50, 50)
   epochs = 100
   ```

4. **Run all cells** to train the model
   - The notebook will automatically:
     - Load and preprocess images
     - Split data into training/validation (80/20)
     - Train the CNN model
     - Apply early stopping
     - Save the best model as `signlanguage3.h5`

5. **Monitor training progress**
   - Watch validation accuracy and loss
   - Training will stop early if no improvement for 20 epochs
   - Best model weights are automatically restored

## 📦 Dependencies

Main libraries required:

```python
streamlit>=1.28.0          # Web interface
opencv-python>=4.8.0       # Image processing
mediapipe>=0.10.0          # Hand tracking
tensorflow>=2.14.0         # Deep learning framework (includes Keras)
numpy>=1.24.0              # Numerical operations
pillow>=10.0.0             # Image handling
```

Install all dependencies:
```bash
pip install streamlit opencv-python mediapipe tensorflow numpy pillow
```

## 🔧 Configuration

### Confidence Threshold
Adjust in `app.py` (line 177):
```python
if confidence > 0.7:  # Change 0.7 to your desired threshold (0.0-1.0)
```

### Camera Resolution
Modify in `app.py` (lines 118-119):
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # Height
```

### Model Input Size
To change model input size, you must:
1. Update `image_size` in `trainmodel.ipynb`
2. Retrain the model
3. Update preprocessing in `app.py` (line 54)

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute
1. **Add new gestures**: Expand the gesture vocabulary
2. **Improve accuracy**: Enhance the model architecture or training process
3. **Optimize performance**: Improve FPS or reduce latency
4. **Better UI**: Enhance the Streamlit interface
5. **Documentation**: Improve guides and tutorials
6. **Bug fixes**: Report and fix issues

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Standards
- Follow PEP 8 style guide for Python code
- Add comments for complex logic
- Test your changes before submitting
- Update documentation as needed

## 📝 License

This project is available for educational and research purposes.

## 🙏 Acknowledgments

- **MediaPipe** by Google for hand tracking technology
- **TensorFlow/Keras** for deep learning framework
- **Streamlit** for web application framework
- Sign language community for gesture references

## 📧 Contact

For questions, suggestions, or issues:
- Open an issue on GitHub
- Check existing issues for solutions
- Refer to the documentation

## 🎯 Future Enhancements

Potential improvements for the project:
- [ ] Add more sign language gestures (expand vocabulary)
- [ ] Support for two-handed gestures
- [ ] Multi-language sign language support (ASL, ISL, etc.)
- [ ] Mobile application deployment
- [ ] Real-time sentence formation
- [ ] Integration with speech synthesis
- [ ] Gesture-to-text translation
- [ ] Video-to-sign language translation
- [ ] Model quantization for edge devices

---

**Made with ❤️ for the sign language community**