import cv2
import numpy as np
import os
import random

# Define directories
os.system("rm -rf fluorescence_images/*")
output_dirs = ["fluorescence_images/cleared", "fluorescence_images/persistent"]
for directory in output_dirs:
    os.makedirs(directory, exist_ok=True)

# Function to create synthetic fluorescence images
def generate_fluorescence_image(image_path, intensity, size=(128, 128), num_spots=30):
    img = np.zeros((size[0], size[1]), dtype=np.uint8)  # Black background
    center = (size[0]//2, size[1]//2)
    radius = 40
    
    # Draw the main glowing spot
    cv2.circle(img, center, radius, (intensity,), -1)  # Glowing spot
    
    # Add random brighter/dimmer spots within the main glowing circle
    for _ in range(num_spots):
        # Random radius and intensity for smaller spots
        spot_radius = random.randint(2, 8)
        spot_intensity = random.randint(intensity - 50, intensity + 50)
        spot_x = random.randint(center[0] - (radius - 8), center[0] + radius)
        spot_y = random.randint(center[1] - (radius - 8), center[1] + radius)
        
        # Ensure the spots are within the main glowing circle
        if (spot_x - center[0])**2 + (spot_y - center[1])**2 <= radius**2:
            cv2.circle(img, (spot_x, spot_y), spot_radius, (spot_intensity,), -1)
    cv2.imwrite(image_path, img)

# Generate sample images for "cleared" (low fluorescence)
for i in range(5000):
    generate_fluorescence_image(f"fluorescence_images/cleared/cleared_{i}.png", intensity=50)

# Generate sample images for "persistent" (high fluorescence)
for i in range(5000):
    generate_fluorescence_image(f"fluorescence_images/persistent/persistent_{i}.png", intensity=190)

# Return file paths to user for verification
os.listdir("fluorescence_images/cleared"), os.listdir("fluorescence_images/persistent")
