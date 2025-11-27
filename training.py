import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
DATA_DIR = 'cats_split'
EPOCHS = 10
FINE_TUNE_LAYERS = 20  # Last N layers to unfreeze

# -------------------
# DATASETS
# -------------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'train'),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'val'),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'test'),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# -------------------
# CLASS WEIGHTS
# -------------------
y_train = np.concatenate([y.numpy() for x, y in train_ds], axis=0)
classes = np.unique(y_train)
class_weights_array = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = {int(c): float(w) for c, w in zip(classes, class_weights_array)}
print("Class weights:", class_weights)

# -------------------
# LOAD PREVIOUS MODEL
# -------------------
model = tf.keras.models.load_model('best_model.keras')
model.summary()

# -------------------
# FIND THE BASE MODEL (MobileNetV2) INSIDE THE MODEL
# -------------------
# MobileNetV2 is usually an instance of Functional API inside the model
base_model = None
for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        if 'mobilenetv2' in layer.name.lower():
            base_model = layer
            break

if base_model is None:
    raise ValueError("Cannot find MobileNetV2 base model inside the loaded model.")

# -------------------
# UNFREEZE LAST N LAYERS
# -------------------
for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
    layer.trainable = False
for layer in base_model.layers[-FINE_TUNE_LAYERS:]:
    layer.trainable = True

# -------------------
# COMPILE WITH LOWER LR
# -------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# -------------------
# CALLBACKS
# -------------------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'best_model_finetuned.keras', save_best_only=True, monitor='val_auc', mode='max'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', patience=5, restore_best_weights=True, mode='max'
    )
]

# -------------------
# TRAIN
# -------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# -------------------
# EVALUATE
# -------------------
test_loss, test_acc, test_auc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
