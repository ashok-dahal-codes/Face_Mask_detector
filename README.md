# Face Mask Detection

A real-time **Face Mask Detection** system built with **Python, OpenCV, TensorFlow, Keras, MobileNetV2, and a pretrained SSD face detector**.

The application uses a webcam to detect faces and classify each detected face as either **Mask** or **No Mask**, displaying the prediction and confidence score directly on the video stream.

##  Overview

This project combines two computer vision models:

1. **SSD Face Detector** — detects faces in each webcam frame.
2. **MobileNetV2-based Mask Classifier** — classifies each detected face as wearing a mask or not.

### Pipeline

```text
Webcam
   ↓
OpenCV SSD Face Detector
   ↓
Face Bounding Boxes
   ↓
Face Cropping & Preprocessing
   ↓
MobileNetV2 Mask Classifier
   ↓
Mask / No Mask
   ↓
Display Result
```

## Features

* Real-time face detection using a webcam
* Detects multiple faces in the same frame
* Face-mask classification
* Displays prediction confidence
* Green bounding box for `Mask`
* Red bounding box for `No Mask`
* Batch prediction for multiple detected faces
* Uses MobileNetV2 for efficient image classification
* Uses pretrained models for faster development

## 🛠️ Technologies Used

* **Python**
* **OpenCV**
* **TensorFlow**
* **Keras**
* **MobileNetV2**
* **NumPy**
* **imutils**
* **SSD Face Detector**

## Project Structure

```text
facemask_detection/
│
├── face_detector/
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
│
├── generate_mask_model.py
│
├── main.py
├── requirements.txt
├── mask_detector.keras
└── venv/
```

> `venv/` should normally **not be committed to GitHub**. Add it to `.gitignore`.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/facemask_detection.git
cd facemask_detection
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For this project, an OpenCV 4.x version is recommended because the SSD face detector uses the older Caffe model format.

Example:

```bash
pip install opencv-python==4.10.0.84
```

## Requirements

Example `requirements.txt`:

```text
tensorflow
opencv-python==4.10.0.84
imutils
numpy
```

You can install everything with:

```bash
pip install -r requirements.txt
```

## Running the Application

Make sure your webcam is connected and your virtual environment is activated.

Run:

```bash
python main.py
```

The application will open the webcam and start detecting faces.

Press:

```text
q
```

to stop the application.

## How It Works

### 1. Face Detection

The project uses an OpenCV DNN implementation of an SSD face detector.

The following files are required:

```text
face_detector/
├── deploy.prototxt
└── res10_300x300_ssd_iter_140000.caffemodel
```

The detector identifies the location of faces in each frame.

### 2. Face Preprocessing

Each detected face is:

* Cropped from the frame
* Converted from BGR to RGB
* Resized to `224 × 224`
* Converted to a NumPy array
* Preprocessed using MobileNetV2 preprocessing

### 3. Mask Classification

The processed face is passed to the mask classification model.

The classifier produces two outputs:

```text
Mask
No Mask
```

The class with the higher probability is displayed on the screen.

## Model Architecture

The mask classifier is based on **MobileNetV2**.

The general architecture is:

```text
Input Image
   │
   ▼
MobileNetV2
(pretrained on ImageNet)
   │
   ▼
Global Average Pooling
   │
   ▼
Dense Layer (128 neurons)
   │
   ▼
Dropout (0.2)
   │
   ▼
Dense Layer (2 neurons)
   │
   ▼
Softmax
   │
   ├── Mask
   └── No Mask
```

The MobileNetV2 base is used as a feature extractor, while the classification layers are responsible for the mask/no-mask classification.

## Important Note About Training

The model architecture alone does **not** train a face-mask detector.

Creating the model with:

```python
model = build_base_model()
model.save("mask_detector.keras")
```

only creates and saves the model. It does not teach the model to recognize masks.

A properly trained model requires a dataset containing examples of:

```text
with_mask/
without_mask/
```

and must be trained using:

```python
model.fit(...)
```

The trained model can then be saved and used by the real-time detection application.

## Limitations

This project is intended primarily as an educational computer vision project.

Performance can vary depending on:

* Training dataset quality
* Lighting conditions
* Camera quality
* Face size
* Face angle
* Occlusions
* Mask type
* Incorrectly worn masks
* Distance from the camera

The displayed probability should not automatically be interpreted as a calibrated real-world confidence score.

## Possible Improvements

Future versions could include:

* Training with a larger and more diverse dataset
* Data augmentation
* Transfer learning and fine-tuning
* Detection of incorrectly worn masks
* Support for video files
* FPS counter
* Better confidence calibration
* Model evaluation with precision, recall, F1-score, and confusion matrix
* GPU acceleration
* Exporting the trained model to ONNX or TensorFlow Lite
* Deployment as a web application

## Output Example

When a face is detected, the application displays a bounding box and prediction:


without mask :
<img width="746" height="601" alt="image" src="https://github.com/user-attachments/assets/7cb7575b-479a-4bb8-b5ad-d3b46383b723" />


with mask :
<img width="758" height="589" alt="image" src="https://github.com/user-attachments/assets/8244e60e-05cd-4910-b783-5daa6c236452" />


## 🔒 Privacy

This application processes webcam frames locally during execution. It does not require uploading webcam frames to an external server.

##

