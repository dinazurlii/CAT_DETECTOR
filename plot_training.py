import numpy as np
import matplotlib.pyplot as plt

try:
    history = np.load('history_finetune.npy', allow_pickle=True).item()
except FileNotFoundError:
    print("History file not found. Run training first!")
    exit()

# Plot accuracy
plt.plot(history['accuracy'], label='train_acc')
plt.plot(history['val_accuracy'], label='val_acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.plot(history['loss'], label='train_loss')
plt.plot(history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
