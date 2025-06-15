from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
from ilc_predictor import predict_image
import tensorflow as tf

# Initialize app
app = Flask(__name__)
CORS(app)

# Upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = tf.keras.models.load_model('CNN_model.h5')  # Change path if needed

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 415

    # Use a unique filename to avoid overwrites
    ext = file.filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{uuid.uuid4()}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(filepath)

    try:
        # Run prediction
        result = predict_image(filepath, model)
        return jsonify(result)
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Failed to process image"}), 500

if __name__ == '__main__':
    app.run(debug=True)
