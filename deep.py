import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from sklearn import svm
from sklearn.metrics import classification_report

# Load the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the CNN architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=64, verbose=1, validation_data=(test_images, test_labels))

# Extract features from the last fully connected layer
feature_extractor = models.Model(inputs=model.input, outputs=model.layers[-2].output)
train_features = feature_extractor.predict(train_images)
test_features = feature_extractor.predict(test_images)

# Train a linear SVM classifier
clf = svm.SVC(kernel='linear')
clf.fit(train_features, train_labels)

# Evaluate the SVM classifier on the test set
svm_predictions = clf.predict(test_features)
print(classification_report(test_labels, svm_predictions))
