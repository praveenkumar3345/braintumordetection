import cv2
import numpy as np
import tensorflow as tf
import sys

# Load the trained model
model = tf.keras.models.load_model("brain_tumor_model.h5")

# Define categories (must match training labels)
CATEGORIES = ["glioma", "meningioma", "notumor", "pituitary"]

# Image size used during training
IMG_SIZE = 128

def preprocess_image(img_path):
    """ Load, convert to grayscale, resize & normalize the image """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to 128x128
    img = img / 255.0  # Normalize
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 1))  # Reshape for model
    return img

def predict_tumor(image_path):
    """ Predict tumor type from image """
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    class_index = np.argmax(prediction)  # Get the highest probability class
    confidence = prediction[0][class_index] * 100  # Get confidence score
    
    print(f"🧠 Predicted Tumor Type: {CATEGORIES[class_index]} ({confidence:.2f}%)")
    
    # Show image with prediction
    original_img = cv2.imread(image_path)
    cv2.putText(original_img, f"{CATEGORIES[class_index]} ({confidence:.2f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Prediction", original_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run prediction
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    predict_tumor(image_path)
