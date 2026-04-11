# Webcam-Object-Detection-and-Seg
Simple program using yolo models with a webcam to identify and segment objects.
# Installation
#### Prereqs
- Python installed
  - Python [3.14.3](https://www.python.org/downloads/release/python-3143/) was used
  - Any other Python versions should work fine

#### Install pip requirements
```
pip install -r requirements.txt
```

# Running code
#### Run examples
- Object detection and segmentation example
  ```
  py object_seg_example.py
  ```
  This is to try out the model and see how well it works with a provided static image.
- Open webcam example
  ```
  py open_webcam_example.py
  ```
  Just to test if you can open your webcam in the program.

#### Run inference using your webcam as input
```
py webcam_object_seg.py
```