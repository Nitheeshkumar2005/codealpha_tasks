import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Create folders
os.makedirs("images", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load MNIST Dataset
data = np.load("dataset/mnist.npz")

X_train = data["x_train"]
y_train = data["y_train"]

X_test = data["x_test"]
y_test = data["y_test"]

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Save Sample Digits Image
plt.figure(figsize=(10, 4))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(y_train[i])
    plt.axis("off")

plt.tight_layout()
plt.savefig("images/sample_digits.png")
plt.close()

print("Sample Digits Image Saved")

# -----------------------------
# Image Preprocessing
# -----------------------------

X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape for CNN
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# -----------------------------
# CNN Model
# -----------------------------

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])

# Compile Model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN Model Summary")
model.summary()

# -----------------------------
# Train Model
# -----------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_data=(X_test, y_test)
)

# -----------------------------
# Evaluate Model
# -----------------------------

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nTest Accuracy:", accuracy)

# -----------------------------
# Save Model
# -----------------------------

model.save("models/handwritten_model.h5")

print("\nModel Saved Successfully")

# -----------------------------
# Accuracy Graph
# -----------------------------

plt.figure(figsize=(6, 4))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Accuracy")
plt.legend()

plt.savefig("images/training_accuracy.png")
plt.close()

print("Accuracy Graph Saved")

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Predictions
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("images/confusion_matrix.png")
plt.close()

print("Confusion Matrix Saved")