# 👋 Sign Language Interpreter - Real-Time Gesture Recognition

A powerful real-time sign language recognition system combining Convolutional Neural Networks (CNN) with MediaPipe hand tracking technology, achieving an impressive **99.6%+ validation accuracy** (reaching 100% at peak performance). This system enables seamless communication by translating sign language gestures into text in real-time.

## 🎯 Overview

This project implements a sophisticated sign language interpreter that uses computer vision and deep learning to recognize hand gestures in real-time. The system captures video input through a webcam, processes hand landmarks using MediaPipe's 21-point hand tracking, and classifies gestures using a trained CNN model. With real-time processing at 30+ FPS and a user-friendly Streamlit web interface, it provides an accessible solution for sign language interpretation.

**Key Highlights:**
- 🎯 **99.6%+ Validation Accuracy** on custom dataset (100% at peak)
- ⚡ **30+ FPS** real-time processing
- 🤖 **Deep CNN Architecture** with 3 convolutional blocks
- 🖐️ **MediaPipe Integration** for precise hand landmark detection
- 🎨 **Streamlit Web Interface** for easy deployment
- 📊 **Custom Dataset** with 5,000+ images across 10 gesture classes

## ✨ Features

### Real-Time Recognition
- Live webcam feed processing with minimal latency
- Instant gesture classification and display
- Visual feedback with bounding boxes and landmarks
- FPS counter for performance monitoring

### MediaPipe Integration
- 21-landmark hand tracking for precise gesture detection
- Robust hand detection across various lighting conditions
- Automatic hand cropping and normalization
- Support for single-hand gesture recognition

### High Accuracy Model
- 99.6%+ validation accuracy (100% at peak performance)
- Confidence filtering (>70%) to reduce false positives
- Trained on 5,003+ custom-collected images
- Grayscale processing for computational efficiency

### User-Friendly Interface
- Clean Streamlit-based web application
- Side-by-side camera feed and processed hand view
- Real-time prediction display with confidence scores
- Easy-to-use start/stop controls

## 🏗️ Project Structure

```
Sign-language-interpreter-improved-/
│
├── app.py                      # Main Streamlit web application
├── signlanguage3.h5            # Trained CNN model (73.5 MB)
├── trainmodel.ipynb            # Model training notebook
├── createdataset.ipynb         # Dataset collection notebook
├── notebook.ipynb              # Experimental/testing notebook
├── test.ipynb                  # Model testing notebook
│
└── dataset2/                   # Training dataset (5,003 images)
    ├── AAROHAN/               # Gesture class 1
    ├── Am/                    # Gesture class 2
    ├── Are/                   # Gesture class 3
    ├── Fine/                  # Gesture class 4
    ├── Hello/                 # Gesture class 5
    ├── How/                   # Gesture class 6
    ├── I/                     # Gesture class 7
    ├── To/                    # Gesture class 8
    ├── Welcome/               # Gesture class 9
    └── You/                   # Gesture class 10
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam for real-time recognition
- GPU recommended for training (CPU works for inference)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Arka077/Sign-language-interpreter-improved-.git
cd Sign-language-interpreter-improved-
```

### Step 2: Install Dependencies
```bash
pip install streamlit
pip install opencv-python
pip install mediapipe
pip install tensorflow
pip install numpy
pip install pillow
```

Or create a `requirements.txt` with:
```
streamlit>=1.28.0
opencv-python>=4.8.0
mediapipe>=0.10.0
tensorflow>=2.13.0
numpy>=1.24.0
pillow>=10.0.0
```

Then install:
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation
Ensure the trained model `signlanguage3.h5` is in the project root directory.

## 🎮 Usage

### Running the Streamlit Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. The application will open in your default web browser (usually at `http://localhost:8501`)

3. Click the **"Start Camera"** button to begin recognition

4. Perform sign language gestures in front of your webcam

5. The system will display:
   - Live camera feed with hand detection boxes
   - Processed hand image (normalized on white background)
   - Predicted gesture label
   - Confidence score (only shown when >70%)

6. Press **'Q'** in the video window or click **"Stop"** to exit

### Tips for Best Results
- Ensure good lighting conditions
- Keep your hand clearly visible in the frame
- Maintain a moderate distance from the camera
- Hold gestures steady for better recognition
- Wait for the green bounding box to appear around your hand

## 🧠 Model Architecture

The model uses a deep Convolutional Neural Network designed for grayscale image classification:

```
Input: 50x50x1 (Grayscale Image)
    ↓
[Conv2D Block 1]
├─ Conv2D(256 filters, 3x3 kernel, ReLU)
├─ BatchNormalization
├─ MaxPooling2D(2x2)
└─ Dropout(0.5)
    ↓
[Conv2D Block 2]
├─ Conv2D(256 filters, 3x3 kernel, ReLU)
├─ BatchNormalization
├─ MaxPooling2D(2x2)
└─ Dropout(0.5)
    ↓
[Conv2D Block 3]
├─ Conv2D(512 filters, 3x3 kernel, ReLU)
├─ BatchNormalization
├─ MaxPooling2D(2x2)
└─ Dropout(0.5)
    ↓
Flatten
    ↓
[Dense Layers]
├─ Dense(512, ReLU) + Dropout(0.5)
├─ Dense(256, ReLU) + Dropout(0.5)
└─ Dense(64, ReLU) + Dropout(0.5)
    ↓
Output: Dense(10, Softmax) → 10 Classes
```

**Architecture Highlights:**
- **3 Convolutional Blocks** with progressively increasing filters (256→256→512)
- **Batch Normalization** after each convolution for training stability
- **Aggressive Dropout (0.5)** to prevent overfitting
- **MaxPooling** for spatial dimension reduction
- **Deep Dense Layers** (512→256→64) for feature extraction

## 📊 Training Details

### Training Configuration
- **Dataset Size**: 5,003 images
  - Training: 4,003 images (80%)
  - Validation: 1,000 images (20%)
- **Batch Size**: 128
- **Input Size**: 50×50 pixels (grayscale)
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Early Stopping**: Patience of 20 epochs, monitoring validation loss

### Training Results
- **Total Epochs Trained**: 56 (stopped early)
- **Best Epoch**: 36
- **Training Accuracy**: 99.63% (epoch 36)
- **Validation Accuracy**: 100% (epoch 36)
- **Validation Loss**: 3.5542e-07 (epoch 36)

### Training Progress Highlights
| Epoch | Training Acc | Validation Acc | Validation Loss |
|-------|-------------|----------------|-----------------|
| 1     | 21.93%      | 10.00%         | 3.3687          |
| 10    | 98.64%      | 18.40%         | 6.9863          |
| 20    | 99.79%      | 99.80%         | 0.0071          |
| 30    | 99.65%      | 99.90%         | 0.0062          |
| **36** | **99.63%** | **100%**      | **3.5542e-07** |
| 40    | 99.73%      | 100%           | 6.8186e-08      |

The model achieved exceptional performance with early stopping restoring weights from epoch 36, demonstrating excellent generalization capabilities.

## 🔍 Preprocessing Pipeline

The system employs a sophisticated preprocessing pipeline to ensure consistent and accurate predictions:

### 1. Hand Detection & Tracking (MediaPipe)
```
Video Frame → RGB Conversion → MediaPipe Hands → 21 Landmarks
```
- Detects hand presence and extracts 21 3D landmarks
- Provides x, y, z coordinates for each landmark point

### 2. Hand Cropping
```
Landmarks → Bounding Box Calculation → Crop with Padding
```
- Calculates min/max x, y from all landmarks
- Adds 40-pixel padding on all sides
- Extracts hand region from frame

### 3. Normalization
```
Cropped Hand → White Background (224x224) → Aspect Ratio Preservation
```
- Places hand on pure white background
- Maintains aspect ratio to prevent distortion
- Centers hand in the frame

### 4. Model Input Preparation
```
Normalized Image → Resize (50x50) → Grayscale → Normalize [0,1] → Reshape
```
- Resizes to model input dimensions (50×50)
- Converts to grayscale (1 channel)
- Normalizes pixel values to range [0, 1]
- Adds batch and channel dimensions: (1, 50, 50, 1)

### 5. Prediction & Confidence Filtering
```
Model Input → CNN Prediction → Softmax Probabilities → Confidence Check (>70%)
```
- Runs inference through trained CNN
- Applies confidence threshold (70%) to reduce false positives
- Displays prediction only when confidence is sufficient

## 📸 Supported Gestures

The system recognizes 10 distinct sign language gestures:

| Class Index | Gesture | Description |
|------------|---------|-------------|
| 0 | **AAROHAN** | Custom/Organization-specific gesture |
| 1 | **Am** | Verb - "am" |
| 2 | **Are** | Verb - "are" |
| 3 | **Fine** | Adjective - "fine/good" |
| 4 | **Hello** | Greeting gesture |
| 5 | **How** | Question word - "how" |
| 6 | **I** | Pronoun - "I/me" |
| 7 | **To** | Preposition - "to" |
| 8 | **Welcome** | Greeting/welcoming gesture |
| 9 | **You** | Pronoun - "you" |

**Example Sentence Formation:**
- "Hello" + "I" + "Am" + "Fine" = "Hello, I am fine"
- "How" + "Are" + "You" = "How are you?"

## 🛠️ Creating Custom Dataset

To collect your own dataset for additional gestures or to improve existing ones:

### Step 1: Open the Dataset Collection Notebook
```bash
jupyter notebook createdataset.ipynb
```

### Step 2: Configure Collection Settings
```python
folder = "YOUR_GESTURE_NAME"  # Set your gesture class name
counter = 0
```

### Step 3: Run Collection Process
1. Execute all cells in the notebook
2. Position your hand in front of the webcam
3. Press **'S'** to save individual images
4. Press **'Q'** to stop collection

### Best Practices for Dataset Collection
- **Variety**: Collect 500+ images per gesture class
- **Lighting**: Vary lighting conditions (bright, dim, natural, artificial)
- **Angles**: Capture gestures from different angles
- **Backgrounds**: Use diverse backgrounds to improve robustness
- **Positioning**: Vary hand position (left, right, center, near, far)
- **Performers**: Include multiple people for better generalization
- **Consistency**: Ensure gesture is performed correctly and consistently

### Dataset Organization
After collection, organize images into folders:
```
dataset2/
├── Gesture1/
│   ├── Image1.jpg
│   ├── Image2.jpg
│   └── ...
├── Gesture2/
│   └── ...
```

## 🎓 Training Your Own Model

### Step 1: Open Training Notebook
```bash
jupyter notebook trainmodel.ipynb
```

### Step 2: Configure Training Parameters
```python
data_dir = "dataset2"          # Your dataset directory
batch_size = 128               # Adjust based on GPU memory
image_size = (50, 50)          # Input dimensions
```

### Step 3: Run Training
Execute all cells sequentially. The training process includes:
- Data loading with 80/20 train-validation split
- Data augmentation (rescaling to [0,1])
- Model compilation with Adam optimizer
- Training with early stopping (patience=20)
- Model saving as `signlanguage3.h5`

### Step 4: Monitor Training
- Watch training/validation accuracy and loss
- Early stopping will activate if validation loss doesn't improve
- Best model weights are automatically restored

### Hyperparameter Tuning Tips
- **Batch Size**: Increase for faster training (requires more GPU memory)
- **Learning Rate**: Default Adam works well; consider reducing if overfitting
- **Dropout**: Adjust from 0.5 if needed (increase for more regularization)
- **Epochs**: Maximum 100 with early stopping (typically stops around 35-40)
- **Data Augmentation**: Can add rotation, zoom, brightness variations

## 📦 Dependencies

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **TensorFlow** | ≥2.13.0 | Deep learning framework for model training and inference |
| **Keras** | (included in TF) | High-level neural network API |
| **MediaPipe** | ≥0.10.0 | Hand tracking and landmark detection |
| **OpenCV** | ≥4.8.0 | Computer vision operations and webcam capture |
| **Streamlit** | ≥1.28.0 | Web application framework |
| **NumPy** | ≥1.24.0 | Numerical computing and array operations |
| **Pillow** | ≥10.0.0 | Image processing library |

### Optional for Training
- **Jupyter Notebook** - For running .ipynb files
- **Matplotlib** - For visualization during training
- **Pandas** - For data manipulation and analysis

### System Requirements

**For Inference (Running the App):**
- CPU: Modern multi-core processor
- RAM: 4GB minimum, 8GB recommended
- Webcam: Any standard USB or built-in webcam
- OS: Windows, macOS, or Linux

**For Training:**
- GPU: NVIDIA GPU with 4GB+ VRAM (recommended)
- RAM: 8GB minimum, 16GB+ recommended
- Storage: 5GB+ free space for dataset and models
- CUDA/cuDNN: For TensorFlow GPU support

## 🤝 Contributing

Contributions are welcome! Here's how you can help improve this project:

### Ways to Contribute
1. 🐛 **Bug Reports**: Open an issue describing the bug with reproduction steps
2. ✨ **Feature Requests**: Suggest new features or improvements
3. 📝 **Documentation**: Improve README, add comments, create tutorials
4. 🎨 **UI Enhancements**: Improve the Streamlit interface
5. 🤖 **Model Improvements**: Experiment with different architectures
6. 📊 **Dataset Expansion**: Add more gesture classes or images

### Contribution Guidelines
1. **Fork the Repository**
   - Click the "Fork" button on the GitHub repository page
   - Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Sign-language-interpreter-improved-.git
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Follow existing code style
   - Add comments for complex logic
   - Test your changes thoroughly

4. **Commit Your Changes**
   ```bash
   git commit -m "Add: Brief description of your changes"
   ```

5. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Include screenshots for UI changes

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings for functions and classes
- Keep functions focused and modular

### Areas for Improvement
- [ ] Add more gesture classes (alphabet, numbers, common phrases)
- [ ] Implement two-hand gesture recognition
- [ ] Add gesture-to-speech conversion
- [ ] Create mobile app version
- [ ] Improve model architecture (try ResNet, EfficientNet)
- [ ] Add data augmentation during training
- [ ] Implement real-time translation to multiple languages
- [ ] Add gesture history/sequence tracking
- [ ] Create API for integration with other applications

## 💻 Tech Stack

### Machine Learning & Computer Vision
- **TensorFlow/Keras** - Deep learning framework for CNN model
- **MediaPipe** - Google's ML solution for hand tracking
- **OpenCV** - Computer vision library for image processing

### Web Application
- **Streamlit** - Python framework for building data apps

### Data Processing
- **NumPy** - Numerical computations and array operations
- **Pillow** - Image manipulation

### Development Tools
- **Jupyter Notebook** - Interactive development environment
- **Python 3.12** - Primary programming language

---

## 📄 License

This project is open source and available for educational and research purposes.

## 👨‍💻 Author

**Arka077**

## 🙏 Acknowledgments

- MediaPipe team for the excellent hand tracking solution
- TensorFlow/Keras community for deep learning tools
- Streamlit for the intuitive web framework
- Sign language community for gesture references

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review documentation thoroughly

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for the sign language community

</div>