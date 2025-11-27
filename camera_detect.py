import cv2
import tensorflow as tf
import numpy as np
import random

# Load model
model = tf.keras.models.load_model("best_model_finetuned.keras")

# Labels
labels = ["cute", "ugly"]

# Pesan random untuk tiap label
cute_msgs = ["imup bangett", "miaw lucu", "kucing gemoy"]
ugly_msgs = ["idih jelek banget", "medium ugly cat"]

# Load Haar cascade untuk deteksi kucing
cat_cascade = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")

# Buka kamera laptop
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Tidak bisa membuka kamera")
    exit()

print("Tekan 'q' untuk keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Tidak bisa menerima frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Deteksi kucing
    cats = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in cats:
        # Crop frame kucing
        cat_crop = frame[y:y+h, x:x+w]
        cat_crop_resized = cv2.resize(cat_crop, (224, 224))  # ukuran model
        cat_crop_resized = cat_crop_resized / 255.0
        cat_crop_resized = np.expand_dims(cat_crop_resized, axis=0)
        
        # Prediksi cute / ugly
        pred = model.predict(cat_crop_resized, verbose=0)
        label_idx = np.argmax(pred)
        confidence = pred[0][label_idx]

        # Pilih pesan random
        if labels[label_idx] == "cute":
            msg = random.choice(cute_msgs)
        else:
            msg = random.choice(ugly_msgs)

        label_text = f"{labels[label_idx]}: {confidence:.2f} - {msg}"
        
        # Gambar kotak & label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Cute/Ugly Cat Detector", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
