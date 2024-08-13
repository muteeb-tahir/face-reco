import scipy.misc
import dlib
import numpy as np

# Initialize the face detector and models
face_detector = dlib.get_frontal_face_detector()

landmark_model_path = './models/shape_predictor_68_face_landmarks.dat'
landmark_predictor = dlib.shape_predictor(landmark_model_path)

recognition_model_path = './models/dlib_face_recognition_resnet_model_v1.dat'
face_encoder_model = dlib.face_recognition_model_v1(recognition_model_path)


def rect_to_tuple(rect):
    """
    Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order.

    :param rect: a dlib 'rect' object
    :return: a plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def tuple_to_rect(rect_tuple):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object.

    :param rect_tuple: Plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(rect_tuple[3], rect_tuple[0], rect_tuple[1], rect_tuple[2])


def trim_rect_to_bounds(rect_tuple, image_shape):
    """
    Ensure a tuple in (top, right, bottom, left) order is within the bounds of the image.

    :param rect_tuple: Plain tuple representation of the rect in (top, right, bottom, left) order
    :param image_shape: Numpy shape of the image array
    :return: A trimmed plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return max(rect_tuple[0], 0), min(rect_tuple[1], image_shape[1]), min(rect_tuple[2], image_shape[0]), max(rect_tuple[3], 0)


def compute_face_distance(face_encodings, face_encoding_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a Euclidean distance
    for each comparison face. The distance indicates how similar the faces are.

    :param face_encodings: List of face encodings to compare
    :param face_encoding_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'face_encodings' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_encoding_to_compare, axis=1)


def load_image(filename, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc.) into a numpy array.

    :param filename: Image file to load
    :param mode: Format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
    :return: Image contents as a numpy array
    """
    image = scipy.misc.imread(filename)

    # Resize the image if it is very large
    if image.shape[0] > 800:
        base_height = 500
        scaling_factor = (base_height / image.shape[0])
        new_width = int(image.shape[1] * scaling_factor)
        image = scipy.misc.imresize(image, (base_height, new_width))
    elif image.shape[1] > 800:
        base_height = 500
        scaling_factor = (base_height / image.shape[1])
        new_height = int(image.shape[0] * scaling_factor)
        image = scipy.misc.imresize(image, (new_height, base_height))

    return image


def detect_raw_face_locations(image, upsample_count=1):
    """
    Returns an array of bounding boxes of human faces in an image.

    :param image: An image (as a numpy array)
    :param upsample_count: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :return: A list of dlib 'rect' objects of found face locations
    """
    return face_detector(image, upsample_count)


def detect_face_locations(image, upsample_count=1):
    """
    Returns an array of bounding boxes of human faces in an image.

    :param image: An image (as a numpy array)
    :param upsample_count: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :return: A list of tuples of found face locations in (top, right, bottom, left) order
    """
    return [trim_rect_to_bounds(rect_to_tuple(face_rect), image.shape) for face_rect in detect_raw_face_locations(image, upsample_count)]


def compute_raw_face_landmarks(image, face_locations=None):
    """
    Computes the facial landmarks for each face detected in the image.

    :param image: An image (as a numpy array)
    :param face_locations: Optionally, provide a list of face locations to check.
    :return: A list of dlib 'full_object_detection' objects of facial landmarks
    """
    if face_locations is None:
        face_locations = detect_raw_face_locations(image)
    else:
        face_locations = [tuple_to_rect(face_loc) for face_loc in face_locations]

    return [landmark_predictor(image, face_loc) for face_loc in face_locations]


def detect_face_landmarks(image, face_locations=None):
    """
    Given an image, returns a dictionary of face feature locations (eyes, nose, etc.) for each face in the image.

    :param image: Image to search
    :param face_locations: Optionally, provide a list of face locations to check.
    :return: A list of dictionaries of face feature locations (eyes, nose, etc.)
    """
    landmarks = compute_raw_face_landmarks(image, face_locations)
    landmarks_as_tuples = [[(point.x, point.y) for point in landmark.parts()] for landmark in landmarks]

    return [{
        "chin": points[0:17],
        "left_eyebrow": points[17:22],
        "right_eyebrow": points[22:27],
        "nose_bridge": points[27:31],
        "nose_tip": points[31:36],
        "left_eye": points[36:42],
        "right_eye": points[42:48],
        "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
        "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]]
    } for points in landmarks_as_tuples]


def encode_faces(image, known_face_locations=None, num_jitters=1):
    """
    Given an image, return the 128-dimension face encoding for each face in the image.

    :param image: The image that contains one or more faces
    :param known_face_locations: Optionally, the bounding boxes of each face if you already know them.
    :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower.
    :return: A list of 128-dimensional face encodings (one for each face in the image)
    """
    raw_landmarks = compute_raw_face_landmarks(image, known_face_locations)
    return [np.array(face_encoder_model.compute_face_descriptor(image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]


def compare_face_encodings(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    return list(compute_face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)
