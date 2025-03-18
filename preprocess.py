import cv2
import os
import numpy as np

# Folder Paths (Fixed)
TRAIN_PATH = r"D:\brain_tumor\archive\Training"
TEST_PATH = r"D:\brain_tumor\archive\Testing"
CATEGORIES = ["glioma", "meningioma", "notumor", "pituitary"]  # Tumor types
IMG_SIZE = 128  # Resize Size

def preprocess_image(img_path):
    """ Load image, convert to grayscale, resize & normalize """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to 128x128
    img = img / 255.0  # Normalize (0-1 range)
    return img

def load_data(data_path):
    """ Load and preprocess dataset """
    data = []
    labels = []

    for category in CATEGORIES:
        folder_path = os.path.join(data_path, category)
        label = CATEGORIES.index(category)  # Assign labels (0,1,2,3)

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = preprocess_image(img_path)

            data.append(img)
            labels.append(label)

    return np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1), np.array(labels)

# Load Training & Testing Data
X_train, y_train = load_data(TRAIN_PATH)
X_test, y_test = load_data(TEST_PATH)

print(f"✅ Preprocessing Completed! Train Data Shape: {X_train.shape}, Test Data Shape: {X_test.shape}")  
import numpy as np

# Save preprocessed data
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("✅ Data saved successfully! Now you can train the model.")
