from joblib import load
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from skimage.feature import local_binary_pattern, hog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from random import sample
import matplotlib.pyplot as plt

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
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Extract features from the image
    lbp_features = extract_lbp_features(image)
    hog_features = extract_hog_features(image)
    combined_features = np.concatenate((lbp_features, hog_features))
    return combined_features

class FaceDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        contents = os.listdir(data_dir)
        self.images = [f for f in contents if f.endswith('.jpg') ]  
        print(self.images)
        self.images.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.data_dir, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return image
    
def plot_image_with_prediction(image, predicted_label):
    plt.imshow(image, cmap='gray')  # Assuming the image is grayscale
    plt.title("Predicted Label: " + str(predicted_label))
    plt.axis('off')
    plt.show()

data_dir = 'C:/Users/jayag/Desktop/ip_project/test_facedataset'
dataset = FaceDataset(data_dir)

num_images_to_show = len(dataset)
selected_images = sample(dataset.images, num_images_to_show)

for image_name in selected_images:
    test_image_path = os.path.join(data_dir, image_name)
    # Preprocess and extract features from the test image
    test_features = preprocess_and_extract_features(test_image_path)
    # Use the loaded model for prediction
    predicted_label = loaded_model.predict([test_features])[0]
    # Print the predicted label
    print("Predicted Label:", predicted_label)
    plot_image_with_prediction(cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE), predicted_label)
