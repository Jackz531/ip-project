from joblib import load
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
import os
from torch.utils.data import Dataset
from random import sample
import matplotlib.pyplot as plt
import pandas as pd

# Define a function to extract LBP features from an image
def extract_lbp_features(image):
    lbp_radius = 3
    lbp_points = 24
    lbp_image = local_binary_pattern(image, lbp_points, lbp_radius, method='uniform')
    hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, lbp_points + 3), range=(0, lbp_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# Define a function to extract HOG features from an image
def extract_hog_features(image):
    hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
    return hog_features

# Load the saved Random Forest model
model_filename = 'random_forest_model.joblib'
loaded_model = load(model_filename)

# Function to preprocess and extract features from a test image
def preprocess_and_extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    lbp_features = extract_lbp_features(image)
    hog_features = extract_hog_features(image)
    combined_features = np.concatenate((lbp_features, hog_features))
    return combined_features

class FaceDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        contents = os.listdir(data_dir)
        self.images = [f for f in contents if f.endswith('.jpg')]
        self.images.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.data_dir, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return image

def plot_image_with_prediction(image, predicted_label):
    plt.imshow(image, cmap='gray')
    plt.title("Predicted Label: " + str(predicted_label))
    plt.axis('off')
    plt.show()

# Function to activate the camera and take photos
def take_photos_and_predict(data_dir, model):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Press 'Space' to take a photo and 'Esc' to exit.")

    image_counter = 0
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow('Camera', frame)

        key = cv2.waitKey(1)

        if key % 256 == 32:
            img_name = f"image_{image_counter}.jpg"
            cv2.imwrite(os.path.join(data_dir, img_name), frame)
            print(f"{img_name} saved.")

            test_features = preprocess_and_extract_features(os.path.join(data_dir, img_name))
            predicted_label = model.predict([test_features])[0]
            predictions.append((image_counter, predicted_label))

            image_counter += 1

        elif key % 256 == 27:
            print("Escape hit, closing the camera.")
            break

    cap.release()
    cv2.destroyAllWindows()

    unique_predictions = list(set(predictions))
    df = pd.DataFrame(unique_predictions, columns=['SlNo', 'Student Name'])
    df.to_csv('predicted_labels.csv', index=False)
    print("CSV file with predicted labels generated.")

# Call the function to take photos and predict
take_photos_and_predict('test_facedataset', loaded_model)

# The rest of the existing code from the user's script
data_dir = os.path.join(os.getcwd(),'test_facedataset')
dataset = FaceDataset(data_dir)

num_images_to_show = len(dataset)
selected_images = sample(dataset.images, num_images_to_show)

for image_name in selected_images:
    test_image_path = os.path.join(data_dir, image_name)
    test_features = preprocess_and_extract_features(test_image_path)
    predicted_label = loaded_model.predict([test_features])[0]
    print("Student Name:", predicted_label)
    plot_image_with_prediction(cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE), predicted_label)
