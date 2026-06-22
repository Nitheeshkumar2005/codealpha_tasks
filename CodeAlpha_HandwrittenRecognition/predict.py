import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model(
    "models/handwritten_model.h5"
)

# Load dataset
data = np.load("dataset/mnist.npz")

X_test = data["x_test"]
y_test = data["y_test"]

# Preprocess
X_test = X_test / 255.0
X_test = X_test.reshape(-1,28,28,1)

# Predict first image
prediction = model.predict(X_test[:1])

predicted_digit = np.argmax(prediction)

print("Actual Digit   :", y_test[0])
print("Predicted Digit:", predicted_digit)