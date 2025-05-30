from flask import Flask, request, render_template, jsonify, send_from_directory, url_for
import numpy as np
import rasterio
import os
from imageio.v2 import imwrite
from tensorflow import keras
from rasterio.enums import Resampling
from PIL import Image, ImageEnhance

app = Flask(__name__)

# Load model
MODEL_PATH = r"models\unet_best_optical24_tf.keras"
segmentation_model = keras.models.load_model(MODEL_PATH)

# Folders for uploads and segmented images
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['SEGMENTED_FOLDER'] = 'static/segmented/'
OUTPUT_FOLDER = 'static/segmented/'  # Ensure OUTPUT_FOLDER points to your static/segmented directory

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SEGMENTED_FOLDER'], exist_ok=True)

# Helper functions
def preprocess_image(file_path):
    with rasterio.open(file_path) as dataset:
        num_bands = dataset.count
        if num_bands < 3:
            raise ValueError(f"Expected at least 3 bands, but found {num_bands}.")
        image = dataset.read([1, 2, 3], out_shape=(3, 256, 256), resampling=Resampling.bilinear)
        image = np.transpose(image, (1, 2, 0)).astype(np.float32) / 255.0
        return np.expand_dims(image, axis=0)

def convert_tiff_to_jpg_with_brightness(tiff_path, jpg_path):
    """Convert TIFF to JPG with brightness enhancement (only for input image)."""
    with rasterio.open(tiff_path) as dataset:
        image = dataset.read([1, 2, 3])
        image = np.transpose(image, (1, 2, 0))
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)
        brightness_enhancer = ImageEnhance.Brightness(pil_image)
        enhanced_image = brightness_enhancer.enhance(1.2)  # Apply brightness only
        enhanced_image.save(jpg_path)
    return jpg_path

def convert_tiff_to_jpg_plain(tiff_path, jpg_path):
    """Convert TIFF to JPG without brightness/contrast (used for segmented image)."""
    with rasterio.open(tiff_path) as dataset:
        image = dataset.read([1, 2, 3])
        image = np.transpose(image, (1, 2, 0))
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)
        pil_image.save(jpg_path)
    return jpg_path

def save_mask_as_tiff(mask, file_path):
    with rasterio.open(
        file_path,
        'w',
        driver='GTiff',
        height=mask.shape[0],
        width=mask.shape[1],
        count=3,
        dtype=np.uint8
    ) as dst:
        for i in range(3):
            dst.write(mask[:, :, i], i + 1)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return jsonify(error="No file uploaded.")
            
            file = request.files['file']
            original_filename = file.filename.split('.')[0]
            tiff_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{original_filename}.tif")
            file.save(tiff_path)

            # Convert TIFF to JPG with brightness (for input display)
            jpg_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{original_filename}.jpg")
            convert_tiff_to_jpg_with_brightness(tiff_path, jpg_path)
            input_image_url = url_for('static', filename=f'uploads/{original_filename}.jpg')

            # Preprocess and predict segmentation
            image = preprocess_image(tiff_path)
            prediction = segmentation_model.predict(image)
            predicted_mask = np.argmax(prediction[0], axis=-1)

            # Generate segmented RGB image
            segmented_image = np.zeros((256, 256, 3), dtype=np.uint8)
            colors = {
                0: [0, 100, 0], 1: [255, 187, 34], 2: [255, 255, 76], 3: [144, 238, 144],
                4: [250, 0, 0], 5: [180, 180, 180], 6:[255,255,255],7: [0, 100, 200],
                8: [139, 69, 19], 9: [0, 255, 0], 10: [220, 220, 220]}


            for class_idx, color in colors.items():
                segmented_image[predicted_mask == class_idx] = color

            # Save segmented image as TIFF
            segmented_tiff_path = os.path.join(app.config['SEGMENTED_FOLDER'], f"{original_filename}_SEGMENTED.tif")
            save_mask_as_tiff(segmented_image, segmented_tiff_path)

            # Convert segmented TIFF to plain JPG (no brightness)
            segmented_jpg_path = os.path.join(app.config['SEGMENTED_FOLDER'], f"{original_filename}_SEGMENTED.jpg")
            convert_tiff_to_jpg_plain(segmented_tiff_path, segmented_jpg_path)
            segmented_image_url = url_for('static', filename=f'segmented/{original_filename}_SEGMENTED.jpg')

            return jsonify({
                "input_image_url": input_image_url,
                "segmented_image_url": segmented_image_url,
                "download_link": url_for('download_file', filename=f"{original_filename}_SEGMENTED.tif"),
                "status_message": "Segmentation Successful!"
            })

        except Exception as e:
            return jsonify(error=str(e))

    return render_template("index.html")

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['SEGMENTED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
