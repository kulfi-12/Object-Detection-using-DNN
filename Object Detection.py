import cv2
import numpy as np

# Helper function to calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = max(0, min(x1 + w1, x2 + w2) - x_intersection)
    h_intersection = max(0, min(y1 + h1, y2 + h2) - y_intersection)

    area_intersection = w_intersection * h_intersection
    area_union = (w1 * h1) + (w2 * h2) - area_intersection

    iou = area_intersection / (area_union + 1e-5)

    return iou

# Load YOLOv3 model and configuration
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class names
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Load an image
image = cv2.imread("sample4.jpeg")

if image is not None:
    # Get image dimensions
    height, width = image.shape[:2]

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Set input blob for the network
    net.setInput(blob)

    # Get output layers' names
    layer_names = net.getUnconnectedOutLayersNames()

    # Run forward pass to get detection results
    outs = net.forward(layer_names)

    # Initialize lists for bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Loop over each of the detection layers
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.8:  # Increased confidence threshold (adjust as needed)
                # Scale the bounding box coordinates back to the original image size
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate coordinates for the top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Initialize variables for evaluation metrics
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Define ground_truth_boxes as a list of bounding boxes
    ground_truth_boxes = []

    # Replace these with actual ground truth bounding box coordinates
    true_x1, true_y1, true_w1, true_h1 = 100, 200, 50, 60
    true_x2, true_y2, true_w2, true_h2 = 250, 300, 40, 80

    # Append the ground truth bounding boxes to the list
    ground_truth_boxes.append((true_x1, true_y1, true_w1, true_h1))
    ground_truth_boxes.append((true_x2, true_y2, true_w2, true_h2))

    # Loop over the remaining boxes
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            # Draw bounding box and label
            color = (0, 255, 0)  # BGR color for the box (here, green)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Initialize a flag to check if this detection has a match with any ground truth box
            matched = False

            # Loop over ground truth boxes and calculate IoU
            for gt_box in ground_truth_boxes:
                IoU = calculate_iou((x, y, w, h), gt_box)

                if IoU < 0.5:
                    true_positives += 1
                    matched = True
                    break  # Stop checking other ground truth boxes once a match is found

            if not matched:
                false_positives += 1

    # Calculate false negatives
    false_negatives = len(ground_truth_boxes) - true_positives

    # Calculate precision, recall, and average precision (AP)
    precision = (true_positives / (true_positives + false_positives))
    recall = true_positives / (true_positives + false_negatives)

    # Display the result
    cv2.imshow("Object Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Report Section 4: Results
    print("Results:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    
    print("Discussion:")
    print("False Positives:", false_positives)
    print("False Negatives:", false_negatives)
    print("...provide analysis and insights here.")

else:
    print("Failed to load the image.")
