import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to convert RGB image to HSI
def rgb_to_hsi(image):
    with np.errstate(divide='ignore', invalid='ignore'):
        b, g, r = cv2.split(image)
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0

        num = 0.5 * ((r - g) + (r - b))
        den = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
        theta = np.arccos(num / (den + 1e-6))

        H = np.where(b <= g, theta, 2 * np.pi - theta)
        H = H / (2 * np.pi)  # Normalize to [0,1]

        num = np.minimum(np.minimum(r, g), b)
        den = r + g + b
        den[den == 0] = 1e-6  # Avoid division by zero
        S = 1 - 3 * num / den

        I = (r + g + b) / 3

        return H, S, I

# Load the image
image_path = 'C:/Users/myash/OneDrive/Pictures/Camera Roll/WIN_20240902_23_10_28_Pro.jpg'
image = cv2.imread(image_path)

# Convert to HSI
H, S, I = rgb_to_hsi(image)

# Convert H, S, I to grayscale based on 8 shades (0-7)
H_gray = (H * 7).astype(np.uint8)
S_gray = (S * 7).astype(np.uint8)
I_gray = (I * 7).astype(np.uint8)

# Plot the images
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Hue Image")
plt.imshow(H_gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Saturation Image")
plt.imshow(S_gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Intensity Image")
plt.imshow(I_gray, cmap='gray')
plt.axis('off')

plt.show()
