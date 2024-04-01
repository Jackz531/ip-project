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

def plot_image_with_prediction(image, predicted_label):
    plt.imshow(image, cmap='gray')  # Assuming the image is grayscale
    plt.title("Predicted Label: " + str(predicted_label))
    plt.axis('off')
    plt.show()

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
        # print(image.shape)
        image.resize(720, 1280)
        return image

X_features = []
y_labels = []

data_dir = 'C:/Users/jayag/Desktop/ip_project/facedataset'
dataset = FaceDataset(data_dir)

num_images_to_show = len(dataset)
selected_images = sample(dataset.images, num_images_to_show)

for image_name in selected_images:
    image_path = os.path.join(data_dir, image_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image.resize(720, 1280)
    name = image_name.split('_')[0]
    lbp_features = extract_lbp_features(image)
    hog_features = extract_hog_features(image)
    combined_features = np.concatenate((lbp_features, hog_features))
    X_features.append(combined_features)
    # Extract label from image name (assuming image names are formatted as "personname_imagename.jpg")
    label = name
    y_labels.append(label)

# Convert lists to numpy arrays
X_features = np.array(X_features)
y_labels = np.array(y_labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)

# Define and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluate the classifier
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# take a random image from the dataset and predict the label
for i in range(10):
    random_idx = np.random.randint(len(dataset))
    random_image = dataset[random_idx]
    random_image.resize(720, 1280)
    random_image_name = dataset.images[random_idx]
    random_lbp_features = extract_lbp_features(random_image)
    random_hog_features = extract_hog_features(random_image)
    random_combined_features = np.concatenate((random_lbp_features, random_hog_features))
    random_combined_features = random_combined_features.reshape(1, -1)
    predicted_label = rf_classifier.predict(random_combined_features)[0]
    plot_image_with_prediction(random_image, predicted_label)

from joblib import dump

# Save the trained Random Forest classifier to a file
model_filename = 'random_forest_model.joblib'
dump(rf_classifier, model_filename)

