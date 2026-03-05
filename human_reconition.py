import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_DIR   = r'D:\Classifier Projects\archive (1)\train'
IMG_SIZE    = 224       # MobileNetV2 standard input size
BATCH_SIZE  = 32
EPOCHS      = 5
MODEL_PATH  = 'human_recognition.keras'
CATS_PATH   = 'categories.txt'

# Threshold: sigmoid > HUMAN_THRESHOLD = non_human, <= HUMAN_THRESHOLD = human
# Raise this above 0.5 if real faces are being misclassified as non-human.
# Try 0.65 or 0.70 if still wrong.
HUMAN_THRESHOLD = 0.65

# =============================================================================
# HELPERS
# =============================================================================
def preprocess_frame(frame_bgr):
    """Apply identical preprocessing to inference frames as used during training."""
    frame_rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE),
                               interpolation=cv2.INTER_LINEAR)
    frame_array   = np.expand_dims(frame_resized.astype('float32'), axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(frame_array)


def build_model():
    """MobileNetV2 transfer learning model for binary classification."""
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base; only train head

    inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x       = base_model(x, training=False)
    x       = GlobalAveragePooling2D()(x)
    x       = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)  # Binary: sigmoid + 1 neuron

    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':

    # ── GPU check ────────────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected: {len(gpus)} GPU(s) available")
        for gpu in gpus:
            print(f"   - {gpu}")
    else:
        print("No GPU detected — using CPU (slower inference)")
    print(f"TensorFlow version: {tf.__version__}\n")

    # ── 1. Build tf.data pipeline with AUTOTUNE prefetching ──────────────────
    print("Loading dataset...")
    AUTOTUNE = tf.data.AUTOTUNE

    # Load raw datasets first to capture class_names before prefetch wraps them
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        INPUT_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )
    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        INPUT_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    # Save class names BEFORE prefetch — prefetch wraps the dataset and drops class_names
    CATEGORIES = train_ds_raw.class_names  # alphabetical: ["human_faces", "non_human_faces"]

    # Now apply AUTOTUNE prefetch for performance
    train_ds = train_ds_raw.prefetch(buffer_size=AUTOTUNE)
    val_ds   = val_ds_raw.prefetch(buffer_size=AUTOTUNE)

    # Verify class index ordering
    print(f"Class names (alphabetical): {CATEGORIES}")
    print(f"  Index 0 = '{CATEGORIES[0]}' (sigmoid <= {HUMAN_THRESHOLD})")
    print(f"  Index 1 = '{CATEGORIES[1]}' (sigmoid >  {HUMAN_THRESHOLD})\n")

    # ── 2. Build and train model ──────────────────────────────────────────────
    print("Building MobileNetV2 transfer learning model...")
    model = build_model()
    model.summary()

    print(f"\nTraining for {EPOCHS} epochs...")
    start_time = time.time()
    model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
    print(f"Training completed in {time.time() - start_time:.1f}s\n")

    # ── 3. Evaluate ───────────────────────────────────────────────────────────
    test_loss, test_accuracy = model.evaluate(val_ds, verbose=0)
    print(f"Validation Accuracy: {test_accuracy * 100:.2f}%")

    # ── 4. Save model and categories ──────────────────────────────────────────
    model.save(MODEL_PATH)
    with open(CATS_PATH, 'w') as f:
        f.write(','.join(CATEGORIES))
    print(f"Model saved to {MODEL_PATH}\n")

    # ── 5. Warm up model (eliminates first-frame latency spike) ───────────────
    dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype='float32')
    _ = model(dummy, training=False)
    print("Model warmed up — starting webcam.\n")

    # ── 6. Live webcam inference at 30+ FPS ───────────────────────────────────
    print("Starting live webcam feed... Press 'q' to quit.")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    # FPS tracking
    fps_counter = 0
    fps_display = 0.0
    fps_timer   = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed — exiting.")
            break

        # ── Detect faces ─────────────────────────────────────────────────────
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        # ── Batch all faces in frame into a single model call ─────────────────
        # This is faster than calling model.predict() once per face.
        # model.predict() has Python-side overhead per call; model() does not.
        if len(faces) > 0:
            batch = np.concatenate([
                preprocess_frame(frame[y:y+h, x:x+w])
                for (x, y, w, h) in faces
            ], axis=0)                                      # shape: (N, 224, 224, 3)

            predictions = model(batch, training=False).numpy()  # single forward pass

            for i, (x, y, w, h) in enumerate(faces):
                confidence       = float(predictions[i][0])
                prediction_index = int(confidence > HUMAN_THRESHOLD)
                category         = CATEGORIES[prediction_index]
                display_conf     = confidence if prediction_index == 1 else 1 - confidence

                color = (0, 255, 0) if category == 'human_faces' else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(
                    frame,
                    f"{category}  {display_conf * 100:.1f}%",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

        # ── FPS counter ───────────────────────────────────────────────────────
        fps_counter += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_timer   = time.time()

        cv2.putText(
            frame,
            f"FPS: {fps_display:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
        )

        cv2.imshow('Human vs Non-Human Classifier (MobileNetV2)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Session ended.")