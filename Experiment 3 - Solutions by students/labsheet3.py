import cv2
import numpy as np
import matplotlib.pyplot as plt

# EXPERIMENT 3
# Spatial Filtering Techniques

# Load the image
img = cv2.imread("image.jpg")

# Check whether image was loaded
if img is None:
    print("ERROR: Image not found!")
    print("Make sure image.jpg is in the same folder as exp3.py")
    exit()

print("Image loaded successfully!")

# Convert BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# LOW-PASS FILTERS

# 1. Gaussian Filter
gaussian = cv2.GaussianBlur(gray, (5, 5), 0)

# 2. Median Filter
median = cv2.medianBlur(gray, 5)

# 3. Average / Mean Filter
average = cv2.blur(gray, (5, 5))

# HIGH-PASS FILTERS

# 4. Laplacian Filter
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

# 5. Sobel X
sobel_x = cv2.Sobel(
    gray,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)

sobel_x = cv2.convertScaleAbs(sobel_x)

# 6. Sobel Y
sobel_y = cv2.Sobel(
    gray,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)

sobel_y = cv2.convertScaleAbs(sobel_y)

# DISPLAY RESULTS

plt.figure(figsize=(14, 8))

# Original Image
plt.subplot(2, 4, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

# Grayscale
plt.subplot(2, 4, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

# Gaussian
plt.subplot(2, 4, 3)
plt.imshow(gaussian, cmap="gray")
plt.title("Gaussian Filter")
plt.axis("off")

# Median
plt.subplot(2, 4, 4)
plt.imshow(median, cmap="gray")
plt.title("Median Filter")
plt.axis("off")

# Average
plt.subplot(2, 4, 5)
plt.imshow(average, cmap="gray")
plt.title("Average Filter")
plt.axis("off")

# Laplacian
plt.subplot(2, 4, 6)
plt.imshow(laplacian, cmap="gray")
plt.title("Laplacian Filter")
plt.axis("off")

# Sobel X
plt.subplot(2, 4, 7)
plt.imshow(sobel_x, cmap="gray")
plt.title("Sobel X")
plt.axis("off")

# Sobel Y
plt.subplot(2, 4, 8)
plt.imshow(sobel_y, cmap="gray")
plt.title("Sobel Y")
plt.axis("off")

plt.tight_layout()
plt.show()

# QUESTIONS AND ANSWERS

# 1. What is spatial filtering? How is it used in digital image processing?
# Answer: Spatial filtering is a technique in which the pixel value of an image is modified
# based on the values of its neighboring pixels.
# It is used for image smoothing, noise reduction, edge enhancement, sharpening,
# and improving important image features.
# Spatial filtering is commonly performed using convolution kernels.

# 2. Differentiate between Low-Pass Filters and High-Pass Filters with suitable examples.
# Answer: Low-pass filters reduce high-frequency components and smooth the image.
# They are mainly used for noise reduction and removing small details.
# Examples of low-pass filters are Average Filter, Gaussian Filter, and Median Filter.
# High-pass filters emphasize high-frequency components such as edges and fine details.
# Examples include Laplacian and Sobel filters.

# 3. Compare Average Filter, Gaussian Filter, and Median Filter based on their working principles and applications.
# Answer: The Average Filter replaces each pixel with the average of its neighboring pixels.
# It provides simple image smoothing but may blur edges.
# The Gaussian Filter uses a Gaussian-weighted kernel and provides smoother and more natural
# noise reduction while preserving important structures better than the Average Filter.
# The Median Filter replaces each pixel with the median value of its neighborhood.
# It is especially useful for removing salt-and-pepper noise while preserving edges.

# 4. Why is the Median Filter particularly effective for removing salt-and-pepper noise?
# Answer: Salt-and-pepper noise appears as random black and white pixels in an image.
# The Median Filter replaces each pixel with the median value of neighboring pixels.
# Since extreme noisy values do not strongly affect the median, the noise can be removed effectively.
# At the same time, important edges are preserved better than with simple averaging.

# 5. Explain the role of convolution kernels in spatial filtering.
# Answer: A convolution kernel is a small matrix of numerical values used to process
# the pixels in an image.
# The kernel is moved across the image and its values are multiplied with corresponding
# pixel values and then combined to produce a new pixel value.
# Different kernels are designed for different purposes such as smoothing, sharpening,
# edge detection, and noise reduction.

# 6. What is the purpose of the Sobel and Laplacian operators in edge detection?
# Answer: The Sobel operator detects image gradients and is commonly used to identify
# edges in horizontal and vertical directions.
# The Laplacian operator uses second-order derivatives to detect rapid changes in intensity.
# Both operators are useful for identifying edges and fine details in an image.
# Sobel generally provides directional edge information, while Laplacian detects edges
# without focusing on only one direction.

# 7. Why are filtering operations considered an essential preprocessing step in computer vision?
# Answer: Filtering operations improve image quality before further computer vision processing.
# They can reduce noise, smooth unwanted variations, enhance edges, and preserve important features.
# Better image quality can improve the performance of tasks such as feature extraction,
# image segmentation, object detection, and object recognition.
# Therefore, filtering is an important preprocessing step in computer vision.

# 8. Discuss the trade-off between image smoothing and edge preservation during filtering.
# Answer: Image smoothing reduces noise and unwanted details, but excessive smoothing can
# remove important edges and fine structures.
# On the other hand, preserving edges helps maintain important object boundaries and features,
# but insufficient smoothing may leave noise in the image.
# Therefore, a suitable filter and kernel size must be selected to achieve a balance
# between noise reduction and edge preservation.

# 9. Mention four real-world applications where spatial filtering techniques are widely used.
# Answer: Spatial filtering is widely used in medical imaging, surveillance systems,
# remote sensing, and autonomous vehicles.
# In medical imaging, filtering helps reduce noise and enhance important structures.
# In surveillance, it can improve image quality and enhance object boundaries.
# In remote sensing and autonomous systems, filtering helps improve images for further analysis,
# detection, and recognition tasks.

# 10. Compare spatial domain filtering with frequency domain filtering in terms of implementation and practical applications.
# Answer: Spatial domain filtering directly modifies image pixels using neighboring pixel values
# and convolution kernels.
# It is relatively simple to implement and is commonly used for smoothing, sharpening,
# noise reduction, and edge detection.
# Frequency domain filtering first transforms the image into the frequency domain,
# usually using the Fourier Transform, and then modifies its frequency components.
# Spatial filtering is often convenient for local image operations, while frequency domain
# filtering is useful for analyzing and controlling different frequency components of an image.