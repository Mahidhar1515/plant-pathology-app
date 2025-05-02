import os
import math
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# -----------------------------------------------------
# 1. Parameters
# -----------------------------------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 10
FINE_TUNE_EPOCHS = 5
INITIAL_LR = 1e-4

train_dir = "dataset/train"
val_dir = "dataset/val"  # optional, can reuse part of training data

# -----------------------------------------------------
# 2. Data Generators
# -----------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

val_generator = val_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print("Train classes:", train_generator.class_indices)

# -----------------------------------------------------
# 3. Build Model
# -----------------------------------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
features = Dense(128, activation='relu', name="features")(x)
x = Dropout(0.5)(features)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=INITIAL_LR),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# -----------------------------------------------------
# 4. Callbacks
# -----------------------------------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint("best_crop_disease_model.h5", monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)
]

# -----------------------------------------------------
# 5. Initial Training
# -----------------------------------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=NUM_EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------------------------------
# 6. Fine-tuning
# -----------------------------------------------------
fine_tune_at = 100
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=INITIAL_LR / 10),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=NUM_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=history.epoch[-1],
    callbacks=callbacks,
    verbose=1
)

# Save final model
model.save("crop_disease_model.h5")
print("[INFO] Final model saved as crop_disease_model.h5")

# -----------------------------------------------------
# 7. Compute Class Centroids
# -----------------------------------------------------
# Use shuffle=False to maintain label order
feature_extractor = Model(inputs=model.input, outputs=model.get_layer("features").output)

# Recreate generator with shuffle=False
centroid_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(x
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

features_all = feature_extractor.predict(centroid_generator, steps=math.ceil(centroid_generator.samples / BATCH_SIZE), verbose=1)
labels_all = centroid_generator.classes

centroids = {}
for class_idx in np.unique(labels_all):
    class_features = features_all[labels_all == class_idx]
    centroids[class_idx] = np.mean(class_features, axis=0)

# -----------------------------------------------------
# 8. Compute OOD Threshold (95th percentile)
# -----------------------------------------------------
all_distances = []
for class_idx in np.unique(labels_all):
    class_features = features_all[labels_all == class_idx]
    dists = np.linalg.norm(class_features - centroids[class_idx], axis=1)
    all_distances.extend(dists)

threshold = np.percentile(all_distances, 95)
print("[INFO] Computed OOD threshold (95th percentile):", threshold)

# -----------------------------------------------------
# 9. Save Centroids & Threshold
# -----------------------------------------------------
with open("centroids_and_threshold.pkl", "wb") as f:
    pickle.dump({"centroids": centroids, "threshold": threshold}, f)

print("[INFO] Saved centroids and threshold to centroids_and_threshold.pkl")
