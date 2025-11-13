## 🧠 Object Detection with SSDLite MobileNetV3

This project demonstrates how to perform **object detection** using **PyTorch** and the **SSDLite320-MobileNetV3-Large** model pre-trained on the COCO dataset.  
It loads an input image or video, runs inference, and visualizes bounding boxes with class labels and confidence scores.

---

### 📁 Project Structure

```bash
project/
│
├── input/ # Folder with input images/videos
│   ├── image_1.jpg
│   ├── image_2.jpg
│   ├── video_1.mp4
│   └── video_2.mp4
├── outputs/ # Folder for saving output images/videos
│   ├── images_1_pred.jpg
│   └── video_1_pred.mp4
├── coco_names.py        # COCO dataset class names
├── detect_image.py      # Main script to run detection on images
├── detect_utils.py      # Contains prediction and visualization functions
├── detect_video.py      # Script to run real-time or video detection
├── model.py             # Loads and prepares the SSD model
└── README.md            # Project documentation
---

### ⚙️ Requirements

To run this project, install Python 3.8+ and the following dependencies:

```bash
pip install torch torchvision pillow opencv-python numpy
```


### 🚀 How to Run (Image Detection)

1. Clone the repository
```bash
git clone https://github.com/SEU_USUARIO/object-detection-ssdlite.git
cd object-detection-ssdlite
```

2. Add an image
Place your test image inside the ```input/``` folder.

3. Run the script
```bash
python main.py -i input/image_1.jpg -t 0.5
```
Parameters:
- ```-i```: Path to the input image (default: ```input/image_1.jpg```)
- ```-t```: Detection threshold (default: ```0.5```)

4. Output
The program:
- Displays the image with bounding boxes and confidence scores.
- Saves the result to the outputs/ folder.

### 🎥 How to Run (Video or Webcam)

To perform detection on a video or webcam:
```bash
python video_detection.py -i input/video_1.mp4 -t 0.5
```

Or use your webcam:
```bash
python video_detection.py -i 'webcam' -t 0.5
```

Press ```q``` to stop the video preview.

### 🧩 File Descriptions

```model.py```

Loads the SSDLite320-MobileNetV3-Large model with COCO weights and sets it to evaluation mode.

```detect_utils.py```

Contains helper functions:
- predict(): Runs the model on an image and returns bounding boxes, labels, and confidence scores.
- draw_boxes(): Draws bounding boxes, class names, and confidence scores on the image.

```detect_image.py```

Runs detection on a single image.

```detect_video.py```

Handles real-time detection on video files or webcam streams.


### 🧾 Example Console Output

```bash
Using device: cuda
Image loaded: input/image_1.jpg
Detections found: 4
 - person (0.94)
 - bicycle (0.82)
 - car (0.67)
 - dog (0.58)
```

### 🧱 Model Details

- **Architecture**: SSDLite320-MobileNetV3-Large
- **Pre-trained on**: COCO dataset (91 classes)
- **Framework**: PyTorch / Torchvision
- **Output**: Bounding boxes, labels, and confidence scores