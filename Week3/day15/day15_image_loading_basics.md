# Day 15 – PIL & OpenCV Basics

## Goals for Today
- Learn how to load, display, and manipulate images using PIL and OpenCV
- Understand the differences between PIL images and NumPy arrays
- Explore image formats, channels, and color models
- Prepare for custom preprocessing in future pipelines
- Start building the image loading module for our age prediction model

---

## Lecture 1 – What Is PIL (Pillow)?

PIL (Python Imaging Library) is used for basic image operations in Python.

```python
from PIL import Image

img = Image.open("example.jpg")
img.show()
print(img.size)         # (width, height)
print(img.mode)         # 'RGB', 'L', etc.
```
- You can:

- Convert to grayscale (img.convert("L"))
- Resize images (img.resize((128, 128)))
- Save images in different formats

## Lecture 2 – Converting Images to NumPy Arrays
```python
- PIL images are not arrays by default. To use them for ML:
import numpy as np

img = Image.open("face.jpg").convert("RGB")
array = np.array(img)
print(array.shape)  # (H, W, 3)
```
- To go back to image:
```python
img_back = Image.fromarray(array)
```
- NumPy lets us manipulate pixel data directly — essential for model input.

## Lecture 3 – Reading with OpenCV

- OpenCV is faster and used in real-time CV systems.

```python
import cv2

img = cv2.imread("face.jpg")           # Loads in BGR format!
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```
- OpenCV can:

- Capture webcam input
- Apply filters
- Work with video streams
- But remember: cv2.imread returns a NumPy array directly.

## Lecture 4 – Image Properties: Channels, Shapes, Modes

- Property	PIL (Image)	OpenCV (np.ndarray)
- Color Format	'RGB'	BGR (default)
- Shape	width x height	height x width x channels
- Grayscale	mode = 'L'	single channel array
- Convert to grayscale:
```python
gray = Image.open("img.jpg").convert("L")  # PIL
gray_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # OpenCV
```

## Lecture 5 – Image Format, Compression & Loss

- JPEG: lossy, small, RGB
- PNG: lossless, supports alpha
- BMP/TIFF: large, uncompressed
- WebP: efficient but less supported
- Be aware: Re-saving JPEGs repeatedly can degrade your image quality — especially important for datasets.

## 👷 Model in Progress – Step 1: Image Loading Module

- Today we started building the image input logic for our age predictor.

```python
from PIL import Image
import numpy as np
import os

def load_image(path, target_size=(128, 128), grayscale=False):
    img = Image.open(path)
    if grayscale:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    img = img.resize(target_size)
    return np.array(img)

# Example:
img = load_image("dataset/person1.jpg")
print(img.shape)  # (128, 128, 3)
```

- Next steps:

- Batch loading multiple images
- Normalize pixel values
- Link to age labels

Reflection

Today I practiced loading and converting images using both PIL and OpenCV. I realized how important it is to standardize image shapes and formats before feeding them into a model. This is the first concrete step toward building our age detection pipeline.

Next Up: Day 16 – Image → NumPy Array (Dataset Loader Continues)