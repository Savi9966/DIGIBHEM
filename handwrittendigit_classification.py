import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Data Preparation
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images to the range [0,1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Expand dimensions for CNN (batch_size, height, width, channels)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Step 2: Model Selection - CNN Architecture
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Step 3: Training the CNN model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Step 4: Model Evaluation
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# Step 5: Prediction on new handwritten digit images
def predict_image(img):
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=(0, -1))  # Expand dimensions
    prediction = model.predict(img)
    return np.argmax(prediction)

# Test prediction
plt.imshow(x_test[0].reshape(28,28), cmap='gray')
prediction = predict_image(x_test[0])
plt.title(f'Predicted Digit: {prediction}')
plt.show()
