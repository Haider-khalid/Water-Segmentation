
import io
import base64
import numpy as np
import tensorflow as tf
import tifffile as tiff
from flask import Flask, request, jsonify, render_template
# import matplotlib
# matplotlib.use('Agg')  # استخدام خلفية Agg بدلاً من TkAgg
import matplotlib.pyplot as plt


app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('Mobilenet_segmentation.keras')

def read_images(image_stream, label_stream, image_height, image_width):
    image = tiff.imread(io.BytesIO(image_stream.read()))
    min_val = np.min(image)
    max_val = np.max(image)
    # تطبيق min-max scaling
    image = (image - min_val) / (max_val - min_val)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    
    mask = tf.io.decode_image(label_stream.read(), channels=1)
    mask = tf.squeeze(mask, axis=-1)
    mask = tf.one_hot(mask, depth=2)
    
    image = tf.image.resize(image, [image_height, image_width])
    mask = tf.image.resize(mask, [image_height, image_width], method='nearest')
    
    return image, mask

def preprocess_image(image_file, mask_file, image_height, image_width):
    image, mask = read_images(image_file, mask_file, image_height, image_width)
    image = tf.expand_dims(image, axis=0)  # Expand dimensions to match model input
    return image, mask

def display_true_predicted_masks_flask(true_mask, predicted_mask):
    if true_mask.shape[-1] == 2:
        true_mask = np.argmax(true_mask, axis=-1)
    if len(predicted_mask.shape) == 3: 
        predicted_mask = np.squeeze(predicted_mask, axis=-1)
    
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 15))  
    axes[0].imshow(true_mask, cmap='gray')
    axes[0].set_title('True Mask', fontsize=28) 
    axes[0].axis('off')

    axes[1].imshow(predicted_mask, cmap='gray')
    axes[1].set_title('Predicted Mask', fontsize=28) 
    axes[1].axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')  
    buf.seek(0)
    plt.close(fig)

    return buf

def get_class_type(predicted_mask):
    water_pixels = np.sum(predicted_mask == 1)
    total_pixels = predicted_mask.size
    water_percentage = (water_pixels / total_pixels) * 100

    if water_percentage > 50:
        return "Water"
    else:
        return "No Water"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files or 'mask' not in request.files:
            return jsonify({'error': 'No image or mask provided'}), 400

        image_file = request.files['image']
        mask_file = request.files['mask']

        if image_file.filename == '' or mask_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        try:
            preprocessed_image, true_mask = preprocess_image(image_file, mask_file, 128, 128)
            predictions = model.predict(preprocessed_image)
            predicted_mask = np.argmax(predictions[0], axis=-1)
            class_type = get_class_type(predicted_mask)
            true_mask = true_mask.numpy()
            plot_buffer = display_true_predicted_masks_flask(true_mask, predicted_mask)
            plot_data = base64.b64encode(plot_buffer.getvalue()).decode('utf-8')
            return render_template('result.html', plot_url=plot_data, class_type=class_type)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
