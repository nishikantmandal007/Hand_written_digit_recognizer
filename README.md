# Digit Recognition using TensorFlow and Keras
This repository contains a simple implementation of a neural network model for recognizing handwritten digits from the MNIST dataset using TensorFlow and Keras. The model architecture consists of a flattening layer followed by two dense layers with ReLU activation functions and a final output layer with softmax activation.
MNIST Digit Classification using TensorFlow and Keras
Introduction
This Python script utilizes TensorFlow and Keras to create a neural network for classifying handwritten digits from the MNIST dataset. The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0 to 9). The neural network is a simple feedforward model with three layers: a flattening layer, followed by two dense layers with ReLU activation, and a final dense layer with softmax activation for classification.

Code Explanation
Load and Preprocess Data
python
Copy code
# Load the MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0
Display Sample Images
python
Copy code
# Show a sample of training images
fig, axes = plt.subplots(2, 10, figsize=(10, 4))
# ... (code for displaying images)
plt.show()
Define Model Architecture
python
Copy code
# Define the model architecture
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
Compile Model
python
Copy code
# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
Train the Model
python
Copy code
# Train the model and track the accuracy history
epochs = 10
history = model.fit(train_images, train_labels, epochs=epochs, verbose=2)
Plot Training Accuracy
python
Copy code
# Plot the training accuracy
plt.plot(history.history['accuracy'])
plt.title('Model Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
Evaluate the Model
python
Copy code
# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
Test with an Image
python
Copy code
# Test with an image and print the prediction
image_index = 0
test_image = test_images[image_index]
# ... (code for displaying the image)
prediction = model.predict(test_image.reshape(1, 28, 28))
predicted_label = prediction.argmax()
print('The result is:', predicted_label)
Model Summary
python
Copy code
# Print the model summary
model.summary()
Results
The model achieves high accuracy on the test set, demonstrating its effectiveness in classifying handwritten digits. The training accuracy graph shows the model's improvement over epochs. The final test accuracy is printed, and the model's prediction for a specific test image is displayed.

This script serves as a comprehensive example of building, training, and evaluating a neural network for digit classification using TensorFlow and Keras.







