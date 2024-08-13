import os
from random import shuffle
from sklearn import neighbors
import pickle
import numpy as np
import pandas as pd

encoded_data_file = './encoded-images-data.csv'
label_encoder_file = 'face_labels.pkl'

# Load the encoded images data
if os.path.isfile(encoded_data_file):
    data_frame = pd.read_csv(encoded_data_file)
else:
    print('\x1b[0;37;41m' + '{} not found'.format(encoded_data_file) + '\x1b[0m')
    quit()

# Load the label encoder
if os.path.isfile(label_encoder_file):
    with open(label_encoder_file, 'rb') as label_file:
        label_encoder = pickle.load(label_file)
else:
    print('\x1b[0;37;41m' + '{} not found'.format(label_encoder_file) + '\x1b[0m')
    quit()

# Convert the data frame to a NumPy array and shuffle it
all_data = np.array(data_frame.astype(float).values.tolist())
shuffle(all_data)

# Split the data into features (X) and labels (y)
X_train = np.array(all_data[:, 1:-1])
y_train = np.array(all_data[:, -1:])

# Initialize the K-Nearest Neighbors classifier
knn_classifier = neighbors.KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree', weights='distance')
knn_classifier.fit(X_train, y_train.ravel())

classifier_file = "./face_recognizer.pkl"

# Backup the existing classifier file if it exists
if os.path.isfile(classifier_file):
    print('\x1b[0;37;43m' + "{} already exists. Creating a backup.".format(classifier_file) + '\x1b[0m')
    os.rename(classifier_file, "{}.bak".format(classifier_file))

# Save the trained classifier and label encoder to a file
with open(classifier_file, 'wb') as clf_file:
    pickle.dump((label_encoder, knn_classifier), clf_file)
print('\x1b[6;30;42m' + "Classifier saved to '{}'".format(classifier_file) + '\x1b[0m')
