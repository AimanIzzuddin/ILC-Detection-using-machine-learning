# test.py
import cv2
import numpy as np

# Define the function to load and preprocess image
def load_and_preprocess_image(image_path, target_size=(50, 50)):
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img_array = img / 255.0  # Normalize
        return img_array
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

# Prediction wrapper function
def predict_image(image_path, model):
    img_array = load_and_preprocess_image(image_path)
    if img_array is None:
        return {"error": "Image loading and preprocessing failed."}

    img_batch = np.expand_dims(img_array, axis=0)  # For prediction
    predictions = model.predict(img_batch, verbose=0)

    prob_normal = float(predictions[0][0])
    prob_cancer = float(predictions[0][1])
    predicted_class = "Cancer" if prob_cancer >= 0.5 else "Normal"

    return {
        "class": predicted_class,
        "normal_prob": prob_normal,
        "cancer_prob": prob_cancer
    }
