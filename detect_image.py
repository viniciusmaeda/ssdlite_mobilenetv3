import torch
import argparse
import cv2
import detect_utils

from PIL import Image
from model import get_model

"""
Object detection on a single image.

This script loads a pre-trained SSD model, runs inference on a given image,
draws bounding boxes with confidence scores, and displays/saves the result.
"""

# ----------------------------
# Parse command-line arguments
# ----------------------------
parser = argparse.ArgumentParser()

# Path to the input image (default: input/image_1.jpg)
parser.add_argument('-i', '--input', default='input/image_1.jpg', 
                    help='Path to the input image')

# Detection threshold for filtering low-confidence predictions
parser.add_argument('-t', '--threshold', default=0.5, type=float,
                    help='Detection confidence threshold (0–1)')

# Parse arguments into a dictionary
args = vars(parser.parse_args())

# ----------------------------
# Setup device and load model
# ----------------------------
# Use GPU if available; otherwise, use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained SSD model on the chosen device
model = get_model(device)

# ----------------------------
# Load and prepare the image
# ----------------------------
# Open the input image using Pillow
image = Image.open(args['input'])

# ----------------------------
# Run object detection
# ----------------------------
# Predict bounding boxes, class names, labels, and confidence scores
boxes, classes, labels, scores = detect_utils.predict(
    image, model, device, args['threshold']
)

# ----------------------------
# Draw bounding boxes and labels
# ----------------------------
# Draw detected objects with class names and confidence values
image = detect_utils.draw_boxes(boxes, classes, labels, image, scores)

# ----------------------------
# Display and save results
# ----------------------------
# Create a unique name for the output file using image name + threshold
save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{''.join(str(args['threshold']).split('.'))}"

# Show the image in a new window
cv2.imshow('Image', image)

# Save the resulting image with bounding boxes to the 'outputs' folder
cv2.imwrite(f"outputs/{save_name}.jpg", image)

# Wait for a key press before closing the display window
cv2.waitKey(0)