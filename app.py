from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# ──────────────────────────────────────────────────────────────────────────────
# 1) Interpreter import: tiny runtime on Linux, full TF on Windows
# ──────────────────────────────────────────────────────────────────────────────
try:
    # on PythonAnywhere (or any Linux host with the vendored wheel)
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # locally on Windows fallback to full TF
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# alias so the rest of the code doesn’t need changing
tflite = Interpreter

# ──────────────────────────────────────────────────────────────────────────────
# 2) Load the TFLite model once at startup (works everywhere)
# ──────────────────────────────────────────────────────────────────────────────
interpreter = tflite(model_path='models/crop_disease_model.tflite')
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_tflite(img_array):
    inp = np.expand_dims(img_array, 0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])
    return preds

# ──────────────────────────────────────────────────────────────────────────────
# 3) Load OOD centroids & threshold (pickle)
# ──────────────────────────────────────────────────────────────────────────────
with open("centroids_and_threshold.pkl", "rb") as f:
    data = pickle.load(f)
centroids     = data["centroids"]
OOD_THRESHOLD = data["threshold"]

# Attempt to load full TF model for feature‐extraction (if .h5 is present & TF installed)
has_full_tf = False
try:
    # this will fail on PythonAnywhere (no TF wheel)
    import tensorflow as tf
    full_model = tf.keras.models.load_model('models/crop_disease_model.h5')
    feature_extractor = tf.keras.Model(
        inputs=full_model.input,
        outputs=full_model.get_layer("features").output
    )
    has_full_tf = True
except Exception:
    feature_extractor = None

# ──────────────────────────────────────────────────────────────────────────────
# 4) Class mapping & suggestions
# ──────────────────────────────────────────────────────────────────────────────
class_indices = {
    0: "apple_diseased",
    1: "apple_healthy",
    2: "corn_diseased",
    3: "corn_healthy",
    4: "grape_diseased",
    5: "grape_healthy",
    6: "sugarcane_diseased",
    7: "sugarcane_healthy",
    8: "wheat_diseased",
    9: "wheat_healthy"
}

suggestions_map = {
    "apple_diseased":   "Prune infected areas and apply a fungicide spray.",
    "apple_healthy":    "Your crop seems healthy. Continue regular monitoring.",
    "corn_diseased":    "Inspect your field for moisture issues and consider fungicide treatment.",
    "corn_healthy":     "Keep up the regular care for optimal growth.",
    "grape_diseased":   "Ensure good air circulation and consider an appropriate treatment.",
    "grape_healthy":    "Your grapes are healthy—monitor for any changes.",
    "sugarcane_diseased":"Check soil nutrients and water management practices.",
    "sugarcane_healthy":"Everything looks great—maintain current practices.",
    "wheat_diseased":   "Consider crop rotation and fungicidal treatment.",
    "wheat_healthy":    "Your wheat crop is healthy. Regular monitoring is advised."
}

# ──────────────────────────────────────────────────────────────────────────────
# 5) Helpers
# ──────────────────────────────────────────────────────────────────────────────
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    arr = image.img_to_array(img)
    arr = arr / 255.0
    return arr

# ──────────────────────────────────────────────────────────────────────────────
# 6) Flask app & routes
# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result     = None
    suggestion = None
    image_file = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join("static", filename)
        file.save(filepath)

        img_array = prepare_image(filepath)

        # 1) TFLite inference
        preds = predict_tflite(img_array)
        idx   = int(np.argmax(preds, axis=1)[0])
        result = class_indices.get(idx, "Unknown")

        # 2) Optional OOD check (only if full TF loaded)
        if has_full_tf:
            feat = feature_extractor.predict(np.expand_dims(img_array,0))[0]
            dists = [np.linalg.norm(feat - c) for c in centroids.values()]
            if min(dists) > OOD_THRESHOLD:
                result = "Unknown"

        suggestion = suggestions_map.get(
            result,
            "No specific suggestion available for unknown crop."
        )
        image_file = filename

    return render_template(
        "index.html",
        result=result,
        suggestion=suggestion,
        image_file=image_file
    )

if __name__ == "__main__":
    app.run(debug=True)
