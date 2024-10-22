# prompt: import mnist from tensorflow and create a neural network to classify the digits also use one of the inputs and evaluate

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert labels to one-hot encoded vectors
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Create the neural network model
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  # Flatten the 28x28 images into a 784-dimensional vector
model.add(Dense(128, activation='relu'))  # Add a hidden layer with 128 neurons and ReLU activation
model.add(Dense(10, activation='softmax'))  # Add an output layer with 10 neurons (for 10 digits) and softmax activation

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test)
print('Test accuracy:', accuracy)


input_image = x_test[50]
prediction = model.predict(input_image.reshape(1, 28, 28))
predicted_digit = tf.argmax(prediction, axis=1).numpy()[0]
print('Predicted digit:', predicted_digit)
print('Actual digit:', tf.argmax(y_test[0], axis=0).numpy())
