import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt

# Initialize file dialog
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

# Read image and detect edges
img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
img = cv2.blur(img,(2,2))

# Region of interest: focus on the road (mask the upper parts or irrelevant areas)
mask = np.zeros_like(img)
height, width = img.shape
# Define a polygon for masking (adjust coordinates as per the road area in the image)
polygon = np.array([[(0, height), (width, height), (width, int(height * 0.6)), (0, int(height * 0.6))]])
cv2.fillPoly(mask, polygon, 255)
masked_img = cv2.bitwise_and(img, mask)

# Tune Canny edge detection thresholds
edges = cv2.Canny(masked_img, 100, 150)
cv2.imshow("Edges",edges)

# Define theta and rho ranges (limit theta to a smaller range that aligns with road angles)
theta = np.deg2rad(np.arange(-50, 50, 1))  # Restrict to a smaller range (for more vertical lines)
diag_len = int(np.sqrt(width**2 + height**2))
rhos = np.arange(-diag_len, diag_len, 1)

# Initialize accumulator
accumulator = np.zeros((len(rhos), len(theta)), dtype=int)

# Perform Hough transform
for y in range(height):
    for x in range(width):
        if edges[y, x]:  # If edge point
            for tht_index, tht in enumerate(theta):
                rho = int(x * np.cos(tht) + y * np.sin(tht))
                rho_index = np.argmin(np.abs(rhos - rho))
                accumulator[rho_index, tht_index] += 1

# Increase threshold to detect prominent lines only
threshold = np.max(accumulator) * 0.6  # Adjust this based on your image (70% of the max)
lines = np.argwhere(accumulator > threshold)  # Get indices of points above threshold

# Convert the lines (rho, theta) back to x, y coordinates and draw them on the image
output_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR for colored lines

for rho_index, tht_index in lines:
    rho = rhos[rho_index]
    tht = theta[tht_index]
    
    a = np.cos(tht)
    b = np.sin(tht)
    x0 = a * rho
    y0 = b * rho
    
    # Extend the line to the image borders
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    
    cv2.line(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw red lines

# Show the image with the detected lines
cv2.imshow('Detected Lines', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Visualize the accumulator
plt.imshow(accumulator, cmap='gray', extent=[np.rad2deg(theta[0]), np.rad2deg(theta[-1]), -diag_len, diag_len], aspect='auto')
plt.xlabel('Theta (degrees)')
plt.ylabel('Rho (pixels)')
plt.title('Hough Transform - Grayscale Heatmap')
plt.show()
