# Workshop 2 : Object-detection-using-web-camera
# Name: Raha Priya Dharshini M
# Reg no.: 212224240124


## Aim
To implement real-time object detection using the YOLO (You Only Look Once) model and OpenCV through a webcam.

## Software Requirements
- Python
- OpenCV
- Ultralytics YOLO
- NumPy
- Matplotlib
- Jupyter Notebook

## Procedure
1. Import the required Python libraries.
2. Install the Ultralytics, OpenCV and Matplotlib libraries.
3. Load the pre-trained YOLOv8 model using `YOLO("yolov8m.pt")`.
4. Open the webcam using `cv2.VideoCapture(0)`.
5. Capture video frames from the webcam.
6. Pass each frame to the YOLO model for object detection with a confidence threshold of 0.40.
7. Draw bounding boxes and object labels using `results[0].plot()`.
8. Display the detected objects using Matplotlib.
9. Release the webcam after processing the frames.

## Code:
```

import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8m.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")

else:
    try:
        for i in range(30):
            ret, frame = cap.read()

            if not ret:
                print("Failed to capture frame.")
                break

            # Perform object detection
            results = model(
                frame,
                conf=0.40,
                verbose=False
            )

            # Draw bounding boxes and labels
            annotated_frame = results[0].plot()

            # Convert BGR to RGB
            annotated_frame = cv2.cvtColor(
                annotated_frame,
                cv2.COLOR_BGR2RGB
            )

            # Display output
            clear_output(wait=True)
            plt.figure(figsize=(10, 6))
            plt.imshow(annotated_frame)
            plt.title("Real-Time Object Detection")
            plt.axis("off")
            plt.show()

    finally:
        cap.release()
        cv2.destroyAllWindows()

print("Object detection completed.")
```

## Output

### 1. Real-Time Object Detection
<img width="1362" height="623" alt="image" src="https://github.com/user-attachments/assets/7a03bccf-a13f-41d5-9970-ebd329a7c9a7" />



## Result
Thus, real-time object detection was implemented successfully using the YOLOv8 model and OpenCV. Objects captured through the webcam were detected and displayed with bounding boxes and confidence scores.
