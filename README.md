# DataAugmentation
A Python library for image data augmentation using Pillow and NumPy.
It provides functions to resize, rotate, add noise, crop images, and convert images to tensor formats, facilitating machine learning model training.

## Features
- Resize images to a fixed size

- Rotate images with fixed or random angles

- Add Gaussian noise to images

- Crop images randomly or by specified coordinates

- Convert images to tensors in HWC (Height, Width, Channels) or CHW (Channels, Height, Width) formats

## Installation
Make sure you have the required dependencies:

pip install pillow numpy
## Usage
Import and initialize the DataAugmentation class by providing the path to your image directory:

```python
from data_augmentation import DataAugmentation

augment = DataAugmentation(directory_path="path/to/images", size_img=255)

resize_image()
#Resize all images in the directory to the specified size (size_img) and save them.


Image_rotation(random=False, quantity=0, min=10, max=340)
#Rotate images by fixed angles (45, 90, 135, 180, 225, 270, 315) by default.
#If random=True, rotate images by quantity random angles between min and max degrees.

Image_noise(sigma)
#Add Gaussian noise with standard deviation sigma to each image.

Image_crop(left=0, upper=0, right=0, lower=0, random=False)
#Crop images either randomly (if random=True) or by the specified coordinates (left, upper, right, lower).

Image_tensorHWC()
#Convert all images to normalized tensors with shape (Height, Width, Channels).
#Returns a list of numpy arrays.

Image_tensorCHW()
#Convert all images to normalized tensors with shape (Channels, Height, Width).
#Returns a list of numpy arrays.
```
# Example

```python
augment = DataAugmentation("images_folder", size_img=128)

augment.resize_image()
augment.Image_rotation(random=True, quantity=5, min=15, max=90)
augment.Image_noise(sigma=10)
augment.Image_crop(random=True)

tensors_hwc = augment.Image_tensorHWC()
tensors_chw = augment.Image_tensorCHW()
```
