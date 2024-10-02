import cv2
import numpy as np

# Load YOLO model and classes
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]

# Use the default camera (camera index 0)
cap = cv2.VideoCapture (0)

# Get the output layer names
output_layers_names = net.getUnconnectedOutLayersNames()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Get image dimensions
    height, width, _ = frame.shape

    # Convert the image to blob to be fed into the network
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Set the input blob for the network
    net.setInput(blob)

    # Forward pass to get the outputs
    outputs = net.forward(output_layers_names)

    # Lists for detected objects, their confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Loop through each output layer and get the detected objects
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] == 'car':
                # Object detected is a car
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)

                # Rectangle coordinates
                x, y = center_x - w // 2, center_y - h // 2

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression to remove overlapping bounding boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes on the frame
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Real-time car Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object and close windows
cap.release()
cv2.destroyAllWindows()