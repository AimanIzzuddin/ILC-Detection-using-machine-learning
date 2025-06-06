import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the function to load and preprocess image
def load_and_preprocess_image(image_path, target_size=(50, 50)):
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, target_size)
        img_array = img / 255.0  # Normalize pixels to [0,1]
        return img_array
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

# Path to your image
image_path = '/kaggle/input/miniddsm2/MINI-DDSM-Complete-JPEG-8/Benign/0029/C_0029_1.LEFT_CC.jpg'

# Load and preprocess
img_array = load_and_preprocess_image(image_path)

if img_array is not None:
    img_batch = np.expand_dims(img_array, axis=0)  # Create batch for prediction
    predictions = model.predict(img_batch, verbose=0)  # Predict

    # Extract prediction probabilities for each class
    prob_normal = predictions[0][0]
    prob_cancer = predictions[0][1]

    predicted_class = "Cancer" if prob_cancer >= 0.5 else "Normal"

    # Plot image and confidence bar
    fig, (ax_img, ax_bar) = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [4, 1]})

    ax_img.imshow(img_array)
    ax_img.axis('off')
    ax_img.set_title(f'Predicted Class: {predicted_class}', fontsize=14)

    classes = ['Normal', 'Cancer']
    probs = [prob_normal, prob_cancer]
    colors = ['green', 'red']

    ax_bar.barh(classes, probs, color=colors)
    ax_bar.set_xlim([0, 1])
    ax_bar.set_xlabel('Prediction Confidence')
    ax_bar.set_title('Class Probabilities')
    for i, v in enumerate(probs):
        ax_bar.text(v + 0.02, i, f"{v:.2f}", color='black', va='center')

    plt.tight_layout()
    plt.show()

else:
    print("Image loading and preprocessing failed.")
