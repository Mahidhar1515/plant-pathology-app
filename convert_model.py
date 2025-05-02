import sys
import types
import json
import h5py
import tensorflow as tf
import os

# ─── Fake keras module fix ───────────────────────────────────────────────────────
if "keras" in sys.modules:
    mod = sys.modules["keras"]
    if not hasattr(mod, "__version__"):
        mod.__version__ = tf.__version__
else:
    fake_keras = types.ModuleType("keras")
    fake_keras.__version__ = tf.__version__
    sys.modules["keras"] = fake_keras

# ─── Paths ───────────────────────────────────────────────────────────────────────
H5_PATH    = os.path.join("models", "best_crop_disease_model.h5")
TFLITE_OUT = os.path.join("models", "plant_disease_model.tflite")

# ─── 1) Patch the H5 to ensure shape is present ───────────────────────────────────
print("Patching H5 to ensure `shape` is present...")
with h5py.File(H5_PATH, "r+") as f:
    mc = f.attrs.get("model_config")
    if mc:
        cfg = json.loads(mc.decode("utf-8")) if isinstance(mc, bytes) else json.loads(mc)
        for L in cfg.get("config", {}).get("layers", []):
            if L.get("class_name") == "InputLayer":
                config = L.get("config", {})
                if "batch_shape" in config:
                    batch_shape = config.pop("batch_shape")
                    if batch_shape and isinstance(batch_shape, list):
                        config["shape"] = batch_shape[1:]
                elif "shape" not in config:
                    # Hardcode default shape if missing (example: MobileNetV2 size)
                    config["shape"] = [224, 224, 3]
        f.attrs.modify("model_config", json.dumps(cfg).encode("utf-8"))
print("✔️ H5 patched with shape")

# ─── 2) Load model using tf.keras ─────────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(H5_PATH)
print("✔️ Model loaded")

# ─── 3) Convert to quantized TFLite model ─────────────────────────────────────────
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def rep_data():
    sample = tf.io.decode_image(
        tf.io.read_file(os.path.join("static", "images", "bgd.jpg")), channels=3
    )
    sample = tf.image.resize(sample, [224, 224])
    sample = tf.expand_dims(sample, 0) / 255.0
    for _ in range(50):
        yield [sample]

converter.representative_dataset    = rep_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type      = tf.uint8
converter.inference_output_type     = tf.uint8

# ─── 4) Convert and Save ──────────────────────────────────────────────────────────
print("Converting to TFLite with INT8 quantization...")
tflite_model = converter.convert()

os.makedirs(os.path.dirname(TFLITE_OUT), exist_ok=True)
with open(TFLITE_OUT, "wb") as f:
    f.write(tflite_model)

print(f"✅ Wrote {TFLITE_OUT} ({len(tflite_model)/(1024*1024):.2f} MB)")
