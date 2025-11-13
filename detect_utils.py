import torchvision.transforms as transforms
import cv2
import numpy as np
import torch

from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

# Create a unique color for each class to visualize detections more clearly.
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

# Define the torchvision image transformations (convert image to tensor).
transform = transforms.Compose([
    transforms.ToTensor(),
])

def predict(image, model, device, detection_threshold):
    # Convert the image to a tensor and move it to the selected device (CPU or GPU).
    # Add a batch dimension (model expects a batch of images).
    image = transform(image).to(device).unsqueeze(0)

    # Run inference without computing gradients to speed up processing.
    with torch.no_grad():
        outputs = model(image)

    # Extract prediction outputs: confidence scores, bounding boxes, and labels.
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_boxes = outputs[0]['boxes'].detach().cpu().numpy()
    pred_labels = outputs[0]['labels'].detach().cpu().numpy()

    # Keep only predictions above the confidence threshold.
    mask = pred_scores >= detection_threshold
    boxes = pred_boxes[mask].astype(np.int32)
    scores = pred_scores[mask]
    labels = pred_labels[mask]

    # Get the class names corresponding to the detected labels.
    pred_classes = [coco_names[i] for i in labels]

    # Return bounding boxes, class names, numeric labels, and confidence scores.
    return boxes, pred_classes, labels, scores


def draw_boxes(boxes, classes, labels, image, scores):
    # Convert the image from PIL (RGB) to OpenCV (BGR) format for visualization.
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)

    # Draw all bounding boxes with their corresponding class and confidence score.
    for i, box in enumerate(boxes):
        # Select a color for each class using modulo to avoid index overflow.
        color = COLORS[labels[i] % len(COLORS)]
        
        # Draw the bounding box around the detected object.
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        
        # Display the class name and confidence score above the box.
        text = f"{classes[i]}: {scores[i]:.2f}"
        cv2.putText(
            image, text, 
            (int(box[0]), int(box[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, color, 2, lineType=cv2.LINE_AA
        )
    # Return the image with all bounding boxes drawn.
    return image
