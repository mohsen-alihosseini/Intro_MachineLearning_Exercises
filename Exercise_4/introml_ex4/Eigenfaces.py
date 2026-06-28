from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


N = 64
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TRAINED_CLASSIFIERS = {}
TRAINED_STANDARDIZATION = {}

# Do not add further imports.
# Implement PCA and feature standardization with NumPy only.
# Do not use sklearn.decomposition.PCA or other pre-built standardization helpers.


def _build_classifier(classifier_type):
    if classifier_type == "logistic":
        return LogisticRegression(max_iter=2000)
    if classifier_type == "gaussian_nb":
        return GaussianNB()
    raise ValueError(f"Unknown classifier type: {classifier_type}")


def _uses_feature_scaling(classifier_type):
    return classifier_type == "logistic"


def _list_class_directories(dataset_root):
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    class_dirs = [path for path in sorted(dataset_root.iterdir()) if path.is_dir()]
    if not class_dirs:
        raise ValueError(
            f"Expected at least one class subdirectory in {dataset_root}. "
            "Use a structure like dataset/class_name/image.png."
        )
    return class_dirs


def create_database_from_folder(dataset_root, image_size=(N, N)):
    """
    Load a local image dataset from class subdirectories.

    Expected structure:
        dataset_root/
            class_a/
                img_01.png
            class_b/
                img_02.png
    """
    labels = []
    train = []
    class_dirs = _list_class_directories(dataset_root)

    target_height = None
    target_width = None
    if image_size is not None:
        target_width, target_height = image_size

    for class_dir in class_dirs:
        image_paths = [
            path for path in sorted(class_dir.iterdir()) if path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not image_paths:
            raise ValueError(f"Class directory contains no supported images: {class_dir}")

        for image_path in image_paths:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {image_path}")

            if image_size is not None:
                img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
            elif target_height is None or target_width is None:
                target_height, target_width = img.shape
            elif (img.shape[1], img.shape[0]) != (target_width, target_height):
                raise ValueError(
                    "All images must share the same size when image_size is None. "
                    f"Expected {(target_width, target_height)}, got {(img.shape[1], img.shape[0])} from {image_path}."
                )

            train.append(img.reshape(-1).astype(np.float64))
            labels.append(class_dir.name)

    if not train:
        raise ValueError(f"No images found below {dataset_root}")

    train = np.asarray(train, dtype=np.float64)
    return np.asarray(labels), train, train.shape[0], target_height, target_width


def calculate_average_face(train):
    """
    Calculate the average image using all training images.
    """
    # raise NotImplementedError("Implement calculate_average_face().")
    return np.mean(train,axis=0)


def calculate_eigenfaces(train, avg, num_eigenfaces):
    """
    Calculate the principal directions of the centered training set using SVD.
    """
    # raise NotImplementedError("Implement calculate_eigenfaces().")
   
    # Center the data by subtracting the average image
    centered = train - avg
    # centered = U * S * V^T
    # V^T has shape (n_samples, n_pixels) with principal components as rows
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    
    # Return first num_eigenfaces principal components as row vectors
    # Vt already has components as rows, so we take the first num_eigenfaces rows
    return Vt[:num_eigenfaces]


def get_feature_representation(images, eigenfaces, avg, num_eigenfaces):
    """
    Project all images into the PCA space spanned by the first num_eigenfaces components.
    """
    # raise NotImplementedError("Implement get_feature_representation().")
    # Center the images
    centered_images = images - avg
    
    # Project onto first num_eigenfaces principal components
    # Take only the first num_eigenfaces rows of eigenfaces
    features = centered_images @ eigenfaces[:num_eigenfaces].T
    
    return features

def calculate_feature_statistics(features):
    """
    Compute the mean and standard deviation of every PCA feature over the training set.

    Standardize every feature by centering it to zero mean
    and rescaling it to unit standard deviation. Implement this with NumPy only.
    """
    # raise NotImplementedError("Implement calculate_feature_statistics().")
    feature_mean = np.mean(features, axis=0)
    feature_std = np.std(features, axis=0)
    
    # If a feature has zero standard deviation, replace with 1 to avoid division by zero
    feature_std[feature_std == 0] = 1.0
    
    return feature_mean, feature_std

def standardize_features(features, feature_mean, feature_std):
    """
    Standardize all features using the previously computed mean and standard deviation.

    Apply the same transformation to the training features and later to every test image.
    """
    # raise NotImplementedError("Implement standardize_features().")
    return (features - feature_mean) / feature_std

def reconstruct_image(img, eigenfaces, avg, num_eigenfaces, h, w):
    """
    Reconstruct an image from the first num_eigenfaces principal components.
    """
    # raise NotImplementedError("Implement reconstruct_image().")
    # Center the image
    centered = img - avg
    
    # Project onto first num_eigenfaces components to get coefficients
    coeffs = centered @ eigenfaces[:num_eigenfaces].T
    
    # Reconstruct: avg + coeffs @ eigenfaces[:k]
    reconstruction = avg + coeffs @ eigenfaces[:num_eigenfaces]
    
    # Reshape back to image dimensions
    return reconstruction.reshape(h, w)

def process_and_train(labels, train, num_images, h, w, classifier_type="logistic", num_eigenfaces=None):
    """
    Compute PCA features and train one classifier on top of them.
    For Logistic Regression, standardize the PCA features with your own helper functions.
    """
    # raise NotImplementedError("Implement process_and_train().")

    # Set default number of eigenfaces
    if num_eigenfaces is None:
        num_eigenfaces = min(50, train.shape[0] - 1)
    
    # 1. Compute average face
    avg = calculate_average_face(train)
    
    # 2. Compute eigenfaces (principal components)
    eigenfaces = calculate_eigenfaces(train, avg, num_eigenfaces)
    
    # 3. Get PCA features for training data
    features = get_feature_representation(train, eigenfaces, avg, num_eigenfaces)
    
    # 4. Standardize features for Logistic Regression
    if classifier_type == "logistic":
        feature_mean, feature_std = calculate_feature_statistics(feigenfaces)
        features_standardized = standardize_features(features, feature_mean, feature_std)
        
        # Store standardization parameters for later use
        TRAINED_STANDARDIZATION[classifier_type] = (feature_mean, feature_std)
        
        # Train Logistic Regression on standardized features
        classifier = LogisticRegression(max_iter=2000)
        classifier.fit(features_standardized, labels)
    else:
        # Gaussian Naive Bayes - no standardization needed
        classifier = GaussianNB()
        classifier.fit(features, labels)
    
    # Store trained classifier
    TRAINED_CLASSIFIERS[classifier_type] = classifier
    
    return eigenfaces, num_eigenfaces, avg

def train_both_classifiers(labels, train, num_images, h, w, num_eigenfaces=None):
    """
    Train Logistic Regression and Gaussian Naive Bayes on the same PCA features.
    For Logistic Regression, standardize the PCA features with your own helper functions.
    """
    # raise NotImplementedError("Implement train_both_classifiers() after process_and_train().")
    # Set default number of eigenfaces
    if num_eigenfaces is None:
        num_eigenfaces = min(50, train.shape[0] - 1)
    
    # 1. Compute average face
    avg = calculate_average_face(train)
    
    # 2. Compute eigenfaces (principal components)
    eigenfaces = calculate_eigenfaces(train, avg, num_eigenfaces)
    
    # 3. Get PCA features for training data
    features = get_feature_representation(train, eigenfaces, avg, num_eigenfaces)
    
    # 4. Train Logistic Regression (with standardization)
    feature_mean, feature_std = calculate_feature_statistics(features)
    features_standardized = standardize_features(features, feature_mean, feature_std)
    
    logistic_classifier = LogisticRegression(max_iter=2000)
    logistic_classifier.fit(features_standardized, labels)
    TRAINED_CLASSIFIERS["logistic"] = logistic_classifier
    TRAINED_STANDARDIZATION["logistic"] = (feature_mean, feature_std)
    
    # 5. Train Gaussian Naive Bayes (no standardization)
    nb_classifier = GaussianNB()
    nb_classifier.fit(features, labels)
    TRAINED_CLASSIFIERS["gaussian_nb"] = nb_classifier
    
    return eigenfaces, num_eigenfaces, avg

def classify_image(img, eigenfaces, avg, num_eigenfaces, h, w, classifier_type="logistic"):
    """
    Predict the class label of one image from its PCA coefficients.
    If Logistic Regression is used, apply the same feature standardization as during training.
    """
    # raise NotImplementedError("Implement classify_image().")
    # 1. Get PCA coefficients for the image
    features = get_feature_representation(img.reshape(1, -1), eigenfaces, avg, num_eigenfaces)
    
    # 2. Standardize for Logistic Regression
    if classifier_type == "logistic":
        feature_mean, feature_std = TRAINED_STANDARDIZATION.get("logistic", (None, None))
        if feature_mean is not None and feature_std is not None:
            features = standardize_features(features, feature_mean, feature_std)
    
    # 3. Get classifier and predict
    classifier = TRAINED_CLASSIFIERS.get(classifier_type)
    if classifier is None:
        raise ValueError(f"No trained classifier found for type: {classifier_type}")
    
    return classifier.predict(features)