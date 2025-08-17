import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Increase resolution (try 1280x720 or 1920x1080 depending on your cam)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO on the frame
    results = model.predict(frame, conf=0.5, verbose=False)

    # Show results
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Webcam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
