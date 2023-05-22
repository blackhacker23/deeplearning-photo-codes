import numpy as np
import tensorflow as tf
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1) / 255.0

# Convert labels to binary
y_train_binary = np.where(y_train < 5, 1, 0)
y_test_binary = np.where(y_test < 5, 1, 0)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train_binary, epochs=5)

# Extract features using the second to last layer of the model
extractor = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
train_features = extractor.predict(x_train)
test_features = extractor.predict(x_test)

# Train an SVM classifier
svm_model = svm.SVC(kernel='rbf')
svm_model.fit(train_features, y_train_binary)

# Test the SVM classifier
svm_predictions = svm_model.predict(test_features)
svm_accuracy = accuracy_score(y_test_binary, svm_predictions)
print("SVM accuracy:", svm_accuracy)
