from ultralytics import YOLO
import cv2

# Load the ONNX model
model = YOLO('/Users/qing/Downloads/best.pt')
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame

        # resize to 640 / 480
        ratio = 640 / 480
        target_h = ratio * frame.shape[0]
        target_w = ratio * frame.shape[1]
        frame = cv2.resize(frame, (int(target_w), int(target_h)))        
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
