import os
import face_recognition_api
import pickle
import numpy as np
import pandas as pd

def load_images_from_directory(image_directory):
    """
    Loads all image files from the specified directory.
    """
    image_files = [files for _, _, files in os.walk(image_directory)][0]
    valid_images = []
    valid_extensions = [".jpg", ".jpeg", ".png"]
    
    for image_file in image_files:
        _, file_extension = os.path.splitext(image_file)
        if file_extension.lower() in valid_extensions:
            valid_images.append(os.path.join(image_directory, image_file))

    return valid_images

classifier_file = 'face_recognizer.pkl'
test_images_directory = './test-images'

encoded_data_file = './encoded-images-data.csv'
data_frame = pd.read_csv(encoded_data_file)
all_data = np.array(data_frame.astype(float).values.tolist())

X_train = np.array(all_data[:, 1:-1])
y_train = np.array(all_data[:, -1:])

# Load the classifier and label encoder
if os.path.isfile(classifier_file):
    with open(classifier_file, 'rb') as clf_file:
        (label_encoder, knn_classifier) = pickle.load(clf_file)
else:
    print('\x1b[0;37;43m' + "Classifier '{}' not found".format(classifier_file) + '\x1b[0m')
    quit()

# Process each image in the test directory
for image_path in load_images_from_directory(test_images_directory):
    # Print a message with the image name
    print('\x1b[6;30;42m' + "===== Analyzing faces in '{}' =====".format(image_path) + '\x1b[0m')

    image = face_recognition_api.load_image_file(image_path)
    face_locations = face_recognition_api.face_locations(image)

    face_encodings = face_recognition_api.face_encodings(image, known_face_locations=face_locations)
    print("Detected {} faces in the image.".format(len(face_encodings)))

    # Calculate the closest distances for each face encoding
    closest_distances = knn_classifier.kneighbors(face_encodings, n_neighbors=1)

    # Determine if the faces are recognized
    faces_recognized = [closest_distances[0][i][0] <= 0.5 for i in range(len(face_locations))]

    # Make predictions with high confidence and label unknown faces
    predictions = [(label_encoder.inverse_transform(int(pred)).title(), location) if recognized else ("Unknown", location) 
                   for pred, location, recognized in zip(knn_classifier.predict(face_encodings), face_locations, faces_recognized)]

    print(predictions)
    print()
