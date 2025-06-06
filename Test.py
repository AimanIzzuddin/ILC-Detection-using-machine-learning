# Directory containing the images
imgs_dir = glob.glob('/kaggle/input/miniddsm2/MINI-DDSM-Complete-JPEG-8/Benign/**/*.jpeg', recursive=True)

# Example image path
image_path = '/kaggle/input/miniddsm2/MINI-DDSM-Complete-JPEG-8/Benign/0029/C_0029_1.LEFT_CC.jpg'

# Define a function to load and preprocess an image
def load_and_preprocess_image(image_path, target_size=(50, 50)):
    try:
        # Load and preprocess the image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format
        img = cv2.resize(img, target_size)  # Resize to your target size
        img_array = img / 255.0  # Normalize pixel values

        return img_array
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

# Load and preprocess the example image
img_array = load_and_preprocess_image(image_path)

if img_array is not None:
    # Create a batch for prediction (even if it's a single image)
    img_batch = np.expand_dims(img_array, axis=0)

    # Make predictions
    predictions = model.predict(img_batch)

    # Assuming your model predicts binary probabilities, you can get the probability for "Cancer" class
    cancer_probability = predictions[0][0]  # Assuming "Cancer" is the first class

    # Get the predicted class label
    predicted_class = "Cancer" if cancer_probability >= 0.5 else "Normal"

    # Plot the image and display the predicted class and probability
    plt.imshow(img)
    plt.title(f'Predicted Class: {predicted_class}\nProbability of Cancer: {cancer_probability:.4f}')
    plt.axis('off')
    plt.show()
else:
    print("Image loading and preprocessing failed.")