import os
from keras.preprocessing import image
import numpy as np
from sklearn.model_selection import train_test_split
import random

def load_images_and_labels(directory, target_size=(150, 150)):
    images = []
    labels = []
    
    # Loop through all files in the directory
    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        
        # Load the image
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        images.append(img_array)
        
        # Assign label based on the filename
        if 'cat' in img_name.lower():
            labels.append(0)  # Label for cat
        elif 'dog' in img_name.lower():
            labels.append(1)  # Label for dog
    
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

# Example usage
directory = '/path/to/your/dataset'
images, labels = load_images_and_labels(directory)
print("Loaded images shape:", images.shape)
print("Loaded labels shape:", labels.shape)