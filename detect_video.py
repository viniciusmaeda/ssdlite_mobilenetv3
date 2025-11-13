import cv2
import torch
import argparse
import time
import detect_utils

from model import get_model

# ----------------------------
# Parse command-line arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Run object detection on a video stream using SSD MobileNetV3.")
parser.add_argument(
    '-i', '--input',
    default='input/video_1.mp4',
    help='Path to the input video file or "webcam" to use the camera.'
)
parser.add_argument(
    '-t', '--threshold',
    default=0.5,
    type=float,
    help='Detection confidence threshold between 0 and 1.'
)
args = vars(parser.parse_args())

# ----------------------------
# Set computation device (GPU or CPU)
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")

# Load the SSD MobileNetV3 model
model = get_model(device)
model.eval()  # Ensure the model is in evaluation mode

# ----------------------------
# Open the video source
# ----------------------------
# If input is "webcam", use camera index 0; otherwise, open the video file
cap = cv2.VideoCapture(0) if args['input'] == 'webcam' else cv2.VideoCapture(args['input'])

# Check if the video source was successfully opened
if not cap.isOpened():
    print('[ERROR] Could not open video. Please check the path or webcam connection.')
    exit()

# ----------------------------
# Get video properties
# ----------------------------
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create an output file name based on the input video and threshold
save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{''.join(str(args['threshold']).split('.'))}"

# Initialize the VideoWriter to save the processed output video
out = cv2.VideoWriter(
    f"outputs/{save_name}.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),  # Codec: MP4
    30,  # FPS for output
    (frame_width, frame_height)
)

# ----------------------------
# Variables for FPS calculation
# ----------------------------
frame_count = 0      # Counts total processed frames
total_fps = 0.0      # Accumulates FPS to compute the average later

# ----------------------------
# Process each frame in the video
# ----------------------------
while cap.isOpened():
    # Read a single frame
    ret, frame = cap.read()
    if not ret:
        break  # Stop if no more frames are available

    # Record the start time to calculate FPS later
    start_time = time.time()

    # Disable gradient computation for faster inference
    with torch.no_grad():
        # Run detection on the current frame
        boxes, classes, labels, scores = detect_utils.predict(
            frame, model, device, args['threshold']
        )

    # Draw bounding boxes and class labels on the frame
    image = detect_utils.draw_boxes(boxes, classes, labels, frame, scores)

    # Record the end time and compute FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)

    # Accumulate FPS for later averaging
    total_fps += fps
    frame_count += 1

    # Overlay FPS text on the frame
    cv2.putText(
        image, f"{fps:.2f} FPS", (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    # Convert the frame from BGR (OpenCV format) to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Show the frame with detections
    cv2.imshow('Detections', image)

    # Write the frame into the output video
    out.write(image)

    # Press 'q' to stop the video manually
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Stopping video early by user request.")
        break

# ----------------------------
# Cleanup and statistics
# ----------------------------
# Release the video source and writer
cap.release()
out.release()
cv2.destroyAllWindows()

# Compute and print the average FPS across all frames
if frame_count > 0:
    avg_fps = total_fps / frame_count
    print(f"[INFO] Average FPS: {avg_fps:.2f}")
else:
    print("[INFO] No frames were processed.")
