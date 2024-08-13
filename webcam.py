import face_recognition_api as face_rec
import cv2
import os
import pickle
import numpy as np
import warnings

# Optimizations to enhance performance:
#   1. Process each video frame at 1/4th resolution but display it in full resolution.
#   2. Detect faces in every alternate frame to improve speed.

# Access the default webcam (webcam #0)
camera_stream = cv2.VideoCapture(0)

# Load the pre-trained face recognizer model
model_file = 'face_recognizer.pkl'
if os.path.isfile(model_file):
    with open(model_file, 'rb') as model_f:
        (label_encoder, classifier) = pickle.load(model_f)
else:
    print('\x1b[0;37;43m' + "Model '{}' not found".format(model_file) + '\x1b[0m')
    quit()

# Initialize variables
detected_faces = []
encoded_faces = []
identified_names = []
process_frame_toggle = True

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    while True:
        # Capture a single frame from the video stream
        ret, frame = camera_stream.read()

        # Downscale the frame to 1/4th size for faster processing
        reduced_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Process every other frame to save computation time
        if process_frame_toggle:
            # Detect faces and compute their encodings in the current frame
            detected_faces = face_rec.face_locations(reduced_frame)
            encoded_faces = face_rec.face_encodings(reduced_frame, detected_faces)

            identified_names = []
            face_predictions = []
            if len(encoded_faces) > 0:
                nearest_neighbors = classifier.kneighbors(encoded_faces, n_neighbors=1)

                high_confidence = [nearest_neighbors[0][i][0] <= 0.5 for i in range(len(detected_faces))]

                # Predict the identities of the faces and filter out low-confidence matches
                face_predictions = [(label_encoder.inverse_transform(int(pred)).title(), loc) if conf else ("Unknown", loc)
                                    for pred, loc, conf in zip(classifier.predict(encoded_faces), detected_faces, high_confidence)]

        process_frame_toggle = not process_frame_toggle

        # Draw boxes and labels on the identified faces
        for name, (top, right, bottom, left) in face_predictions:
            # Scale up the face locations to match the original frame size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Add a label with the name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the processed video frame
        cv2.imshow('Face Recognition', frame)

        # Press 'q' to exit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    camera_stream.release()
    cv2.destroyAllWindows()
