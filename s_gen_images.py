import cv2
import numpy as np
import os
import random

# Define directories
os.system("rm -rf s_images/*")
output_dirs = ["s_images/spores", "s_images/not_spores"]
for directory in output_dirs:
    os.makedirs(directory, exist_ok=True)

# Function to create synthetic sample images
def generate_s_image(image_path, intensity, size=(512, 512), num_spots=0, is_spore=False):
    img = np.zeros((size[0], size[1]), dtype=np.uint8)  # Black background
    center = (size[0]//2, size[1]//2)
    radius = 160
    
    # Draw the main glowing spot
    cv2.circle(img, center, radius, (intensity,), -1)  # Glowing spot

    num_spots = random.randint(0, 100)
    for _ in range(num_spots):
        shape_type = random.choice(["circle", "lentil", "asterisk"]) if is_spore else "circle"

        # Random position inside the main glowing area
        spot_x = random.randint(center[0] - (radius - 8), center[0] + radius)
        spot_y = random.randint(center[1] - (radius - 8), center[1] + radius)
        
        if (spot_x - center[0])**2 + (spot_y - center[1])**2 <= radius**2:
            if shape_type == "circle":
                # Regular small circular spots
                spot_radius = random.randint(2, 18)
                spot_intensity = random.randint(intensity, intensity + 80)
                cv2.circle(img, (spot_x, spot_y), spot_radius, (spot_intensity,), -1)
            
            elif shape_type == "lentil":
                # Lentil/ellipse shape
                major_axis = random.randint(2, 7)
                minor_axis = random.randint(1, 2)
                angle = random.randint(0, 180)
                spot_intensity = random.randint(intensity + 5, intensity + 100)
                cv2.ellipse(img, (spot_x, spot_y), (major_axis, minor_axis), angle, 0, 360, (spot_intensity,), -1)
            
            elif shape_type == "asterisk":
                # Asterisk shape using lines
                spot_intensity = random.randint(intensity + 2, intensity + 100)
                for angle in range(0, 180, 30):  # Lines at different angles
                    x1 = int(spot_x + random.randint(2, 6) * np.cos(np.radians(angle)))
                    y1 = int(spot_y + random.randint(2, 6) * np.sin(np.radians(angle)))
                    x2 = int(spot_x - random.randint(2, 6) * np.cos(np.radians(angle)))
                    y2 = int(spot_y - random.randint(2, 6) * np.sin(np.radians(angle)))
                    cv2.line(img, (x1, y1), (x2, y2), (spot_intensity,), 1)

    cv2.imwrite(image_path, img)

# Generate images for "spores" with extra shapes
for i in range(500):
    generate_s_image(f"s_images/spores/spores_{i}.png", intensity=210, is_spore=True)

# Generate images for "not_spores" (only circles)
for i in range(500):
    generate_s_image(f"s_images/not_spores/not_spores_{i}.png", intensity=210, is_spore=False)

# Return file paths to user for verification
os.listdir("s_images/spores"), os.listdir("s_images/not_spores")
