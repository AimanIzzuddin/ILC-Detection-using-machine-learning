from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- ADD THIS
import os
from ilc_predictor import predict_image
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # <-- ADD THIS LINE
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = tf.keras.models.load_model('CNN_model.h5')  # Replace with your model path

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    result = predict_image(filepath, model)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
