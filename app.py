# app.py
# pip install flask-cors

import os
import io
import json
import numpy as np 
from flask import Flask, request, jsonify, url_for # Add url_for here
from flask_cors import CORS # For CORS support (optional, if needed)
from PIL import Image
from annoy import AnnoyIndex

# Import ONE of the TFLite interpreter options based on requirements.txt
# Option A: Using full TensorFlow
import tensorflow as tf
Interpreter = tf.lite.Interpreter
# Option B: Using TFLite Runtime (preferred for deployment)
# import tflite_runtime.interpreter as tflite
# Interpreter = tflite.Interpreter

# --- Configuration ---
# Determine base directory reliably
BASE_DIR = os.curdir.__str__()
print("Base Directory:", BASE_DIR)
MODEL_DIR = os.path.join(BASE_DIR, 'model')
print("Model Directory:", MODEL_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
print("Data Directory:", DATA_DIR)
# Use the ENCODER TFLite model
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, 'quantized\model_quantized_encoder.tflite')
# FULL_MODEL_PATH = os.path.join(MODEL_DIR, 'full_model\AutoSimNet-V1.h5') # Full model path (if needed)	
# NOTE: Ensure the model is in the correct directory as per your structure


# Annoy index file saved from your notebook
ANNOY_INDEX_PATH = os.path.join(DATA_DIR, 'fashion-product-images-small-annoy-embeddings.ann')
# JSON mapping Annoy ID (as string) to image filename
ITEM_MAPPING_PATH = os.path.join(DATA_DIR, 'item_mapping-fashion-product-images-small.json')

# --- Load TFLite Encoder Model ---
try:
    interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape'] # Expected input shape, e.g., [1, 64, 64, 1]
    input_height = input_shape[1]
    input_width = input_shape[2]
    # Embedding dimension is the output shape (after flatten)
    embedding_dim = output_details[0]['shape'][-1]

    print("✅ TFLite ENCODER model loaded successfully.")
    print(f"   Input Shape: {input_shape}")
    print(f"   Output Shape (Embedding): {output_details[0]['shape']}")

except Exception as e:
    print(f"❌ Error loading TFLite model: {e}")
    exit()

# --- Load Annoy Index and Mapping ---
try:
    # Annoy uses 'angular' for cosine similarity, 'euclidean', etc.
    # Ensure 'f' matches the embedding dimension from the TFLite model output
    annoy_index = AnnoyIndex(embedding_dim, 'angular') # Or 'euclidean' if used in notebook
    annoy_index.load(ANNOY_INDEX_PATH)
    print(f"✅ Annoy index loaded successfully with {annoy_index.get_n_items()} items.")

    with open(ITEM_MAPPING_PATH, 'r') as f:
        item_mapping = json.load(f) # Loads {"annoy_id_str": "filename.jpg", ...}
    print(f"✅ Item mapping loaded successfully with {len(item_mapping)} entries.")

except FileNotFoundError:
    print(f"❌ Error: Annoy index ({ANNOY_INDEX_PATH}) or item mapping ({ITEM_MAPPING_PATH}) not found.")
    print("   Make sure you have generated and placed these files.")
    exit()
except Exception as e:
    print(f"❌ Error loading Annoy index or mapping: {e}")
    exit()


# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app) 
# --- Helper Functions ---

def preprocess_image(image_bytes, target_height, target_width):
    """Loads, resizes (grayscale), and preprocesses image bytes for TFLite model."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('L') # Convert to Grayscale ('L')
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS) # Use a high-quality resampler

        # Convert to numpy array, normalize to [0, 1], and add batch/channel dimensions
        img_array = np.array(img, dtype=np.float32) / 255.0
        input_data = np.expand_dims(img_array, axis=[0, -1]) # Shape: (1, height, width, 1)

        # Verify final shape matches model input details
        if input_data.shape != tuple(input_details[0]['shape']):
             print(f"⚠️ Warning: Preprocessed image shape {input_data.shape} differs from model expected shape {input_details[0]['shape']}")
             # Handle potential mismatch if necessary, e.g., raise error or attempt reshape

        # Check input type if needed (e.g., for quantized models)
        # input_type = input_details[0]['dtype']
        # input_data = input_data.astype(input_type)

        return input_data
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None

def get_embedding(input_data):
    """Runs inference using the loaded TFLite ENCODER model."""
    try:
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke() # Run inference
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data # This is the embedding vector (shape should be [1, embedding_dim])
    except Exception as e:
        print(f"Error during TFLite inference: {e}")
        return None

def find_similar_images_annoy(query_embedding, num_results=10):
    """Finds similar images using the Annoy index."""
    if annoy_index is None or item_mapping is None or query_embedding is None:
        return []

    try:
        # Annoy expects a 1D vector for search
        query_vector = query_embedding.flatten()

        # Get nearest neighbor IDs from Annoy.
        # include_distances=False by default. Set True if needed.
        # search_k=-1 uses default search precision (adjust if needed)
        neighbor_ids = annoy_index.get_nns_by_vector(query_vector, num_results + 1, search_k=-1) # +1 in case query is in index

        # Map Annoy IDs back to filenames using the loaded mapping
        results = []
        for annoy_id in neighbor_ids:
             # Annoy IDs are integers, mapping keys are strings
            annoy_id_str = str(annoy_id)
            if annoy_id_str in item_mapping:
                # Optional: You might want to skip if the result is the exact query image
                # This requires knowing the ID of the query image if it exists in the index
                # if annoy_id != query_id_if_known:
                 results.append(item_mapping[annoy_id_str])

        # Limit to desired number of results (in case query was found and skipped)
        return results[:num_results]

    except Exception as e:
        print(f"Error during Annoy search: {e}")
        return []


# --- API Endpoint ---
@app.route('/recommend', methods=['POST'])
def recommend():
    """API endpoint for image recommendation using TFLite Encoder and Annoy."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            image_bytes = file.read()

            # 1. Preprocess the uploaded image
            input_data = preprocess_image(image_bytes, input_height, input_width)
            if input_data is None:
                return jsonify({"error": "Failed to preprocess image"}), 500

            # 2. Get embedding for the uploaded image using TFLite Encoder
            query_embedding = get_embedding(input_data) # Shape: (1, embedding_dim) 
            if query_embedding is None:
                 return jsonify({"error": "Failed to get embedding from model"}), 500

            # 3. Find similar images using Annoy
            recommendations = find_similar_images_annoy(query_embedding, num_results=18)    # Adjust num_results as needed
            if not recommendations:
                return jsonify({"error": "No similar images found"}), 404

            recommendation_urls = [url_for('static', filename=f'fashion-product-images-small/images/{filename}') for filename in recommendations]
            return jsonify({"recommendations": recommendation_urls}) # Return URLs

        except Exception as e:
            print(f"Error processing request: {e}")
            return jsonify({"error": "Internal server error processing the image"}), 500
    else:
         return jsonify({"error": "Invalid file"}), 400


# --- Health Check Endpoint ---
@app.route('/', methods=['GET'])
def health_check():
    """Basic endpoint to check if the server is running."""
    # Optionally check if model and index are loaded
    status = "ok" if (interpreter and annoy_index and item_mapping) else "error"
    return jsonify({"status": status, "message": "API is running"})


# --- Main Execution ---
if __name__ == '__main__':
    # Runs the Flask development server (for local testing only)
    app.run(debug=True, host='0.0.0.0', port=5000)

 