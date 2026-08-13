import os
import shutil
import tempfile

import cv2
import imutils
import numpy as np

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream


# --------------------------------------------------
# Configuration
# --------------------------------------------------

FACE_CONFIDENCE = 0.5
FRAME_WIDTH = 600

PROTOTXT_PATH = "face_detector/deploy.prototxt"
WEIGHTS_PATH = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
MASK_MODEL_PATH = "mask_detector.model"


# --------------------------------------------------
# Load mask model with Keras 3 compatibility
# --------------------------------------------------

def resolve_mask_model_path():
    candidate_paths = [
        "mask_detector.model",
        "mask_detector.h5",
        "mask_detector.keras",
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Mask model not found. Place the trained model as mask_detector.model, "
        "mask_detector.h5, or mask_detector.keras in the project folder."
    )


def load_mask_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    extension = os.path.splitext(model_path)[1].lower()

    if extension in {".h5", ".keras"}:
        return load_model(model_path)

    try:
        import h5py

        with h5py.File(model_path, "r"):
            pass

        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "mask_detector_legacy.h5")
        shutil.copy2(model_path, temp_path)
        return load_model(temp_path)

    except Exception:
        return load_model(model_path)


# --------------------------------------------------
# Detect faces and predict mask
# --------------------------------------------------

def detect_and_predict_mask(frame, face_net, mask_net):
    """
    Detect faces in a frame and predict whether
    each detected face is wearing a mask.
    """

    (h, w) = frame.shape[:2]

    # Create blob for face detector
    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0)
    )

    # Detect faces
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    locations = []

    # Process every detected face
    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        # Ignore weak detections
        if confidence < FACE_CONFIDENCE:
            continue

        # Get bounding box
        box = detections[0, 0, i, 3:7] * np.array(
            [w, h, w, h]
        )

        (start_x, start_y, end_x, end_y) = box.astype("int")

        # Keep coordinates inside frame
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(w - 1, end_x)
        end_y = min(h - 1, end_y)

        # Extract face
        face = frame[start_y:end_y, start_x:end_x]

        # Skip invalid face regions
        if face.size == 0:
            continue

        # BGR -> RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Resize to MobileNetV2 input size
        face = cv2.resize(face, (224, 224))

        # Convert to array
        face = img_to_array(face)

        # Apply MobileNetV2 preprocessing
        face = preprocess_input(face)

        faces.append(face)
        locations.append(
            (start_x, start_y, end_x, end_y)
        )

    # No faces detected
    if len(faces) == 0:
        return locations, []

    # Convert faces to NumPy array
    faces = np.array(faces, dtype="float32")

    # Predict all faces together
    predictions = mask_net.predict(
        faces,
        batch_size=32,
        verbose=0
    )

    return locations, predictions


# --------------------------------------------------
# Load models
# --------------------------------------------------

print("[INFO] Loading face detector...")

face_net = cv2.dnn.readNet(
    PROTOTXT_PATH,
    WEIGHTS_PATH
)

print("[INFO] Loading mask detector...")

try:
    mask_model_path = resolve_mask_model_path()
    mask_net = load_mask_model(mask_model_path)
except FileNotFoundError as exc:
    print(f"[ERROR] {exc}")
    raise SystemExit(1)

print("[INFO] Models loaded successfully.")


# --------------------------------------------------
# Start webcam
# --------------------------------------------------

print("[INFO] Starting video stream...")
print("[INFO] Press 'q' to quit.")

video_stream = VideoStream(src=0).start()


try:

    while True:

        # Read frame
        frame = video_stream.read()

        if frame is None:
            print("[ERROR] Could not read frame.")
            break

        # Resize frame
        frame = imutils.resize(
            frame,
            width=FRAME_WIDTH
        )

        # Detect faces and predict masks
        locations, predictions = detect_and_predict_mask(
            frame,
            face_net,
            mask_net
        )

        # Draw results
        for (box, prediction) in zip(
            locations,
            predictions
        ):

            (start_x, start_y, end_x, end_y) = box

            # Prediction format:
            # [mask_probability, without_mask_probability]
            mask_probability = prediction[0]
            without_mask_probability = prediction[1]

            # Determine class
            if mask_probability > without_mask_probability:
                label = "Mask"
                probability = mask_probability
                color = (0, 255, 0)

            else:
                label = "No Mask"
                probability = without_mask_probability
                color = (0, 0, 255)

            # Create label
            text = f"{label}: {probability * 100:.2f}%"

            # Draw label
            cv2.putText(
                frame,
                text,
                (start_x, max(20, start_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            # Draw bounding box
            cv2.rectangle(
                frame,
                (start_x, start_y),
                (end_x, end_y),
                color,
                2
            )

        # Display frame
        cv2.imshow(
            "Face Mask Detection",
            frame
        )

        # Press Q to quit
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


finally:

    # Cleanup
    print("[INFO] Stopping video stream...")

    video_stream.stop()

    cv2.destroyAllWindows()

    print("[INFO] Program terminated.")
    