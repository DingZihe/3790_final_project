    # coding:utf-8
import os
import numpy as np
import cv2
from numpy import random
from shutil import copyfile
from pylab import mpl
import string
import random as rnd

mpl.rcParams['font.sans-serif'] = ['SimHei']


# Convert image to vector
def img2vector(image_path):
    if not os.path.exists(image_path):
        print(f"File does not exist: {image_path}")
        return None
    img = cv2.imread(image_path, 0)  # Read image
    if img is None:
        print(f"Failed to read image: {image_path}")
        return None
    img = cv2.equalizeHist(img)  # Histogram equalization
    rows, cols = img.shape
    imgVector = np.reshape(img, (1, rows * cols))  # Flatten the image into a vector
    return imgVector


# Load face images from a specified directory
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img = img2vector(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
                labels.append(filename)
    if len(images) > 0:
        return np.vstack(images), labels
    else:
        return np.array(images), labels


# Define PCA algorithm
def PCA(data, r):
    data = np.float32(np.mat(data))
    rows, cols = np.shape(data)
    data_mean = np.mean(data, 0)  # Mean along columns
    A = data - np.tile(data_mean, (rows, 1))  # Subtract mean from all samples to get A
    C = A * A.T  # Get the covariance matrix
    D, V = np.linalg.eig(C)  # Obtain eigenvalues and eigenvectors of the covariance matrix
    V_r = V[:, 0:r]  # Take the first r eigenvectors along columns
    V_r = A.T * V_r  # Transition small matrix eigenvectors to large matrix eigenvectors
    for i in range(r):
        V_r[:, i] = V_r[:, i] / np.linalg.norm(V_r[:, i])  # Normalize the eigenvectors

    final_data = A * V_r
    return final_data, data_mean, V_r


# Generate a random English name
def generate_random_name(length=8):
    letters = string.ascii_lowercase
    return ''.join(rnd.choice(letters) for i in range(length))


# Face recognition to find similar faces
def find_similar_faces(image_folder, output_folder, r=30):
    # Load all images from the folder
    images, labels = load_images_from_folder(image_folder)

    if images.size == 0:
        print(f"No valid images found in the folder: {image_folder}")
        return

    # Create a new folder named "人脸识别" in the output folder
    face_recognition_folder = os.path.join(output_folder, "人脸识别")
    if not os.path.exists(face_recognition_folder):
        os.makedirs(face_recognition_folder)

    # Use the first image as the target image
    target_image = images[0]
    target_label = labels[0]

    # Use the rest of the images for comparison
    comparison_images = images[1:]
    comparison_labels = labels[1:]

    # Combine target image with images for PCA
    all_images = np.vstack((target_image, comparison_images))

    # Apply PCA
    data_pca, data_mean, V_r = PCA(all_images, r)

    # Project target image and other images onto PCA space
    target_pca = data_pca[0]
    images_pca = data_pca[1:]

    # Calculate Euclidean distances
    distances = np.linalg.norm(images_pca - target_pca, axis=1)

    # Set a threshold to find similar images (tune this value as needed)
    threshold = np.mean(distances) - np.std(distances)  # Adjust threshold
    similar_faces_indices = np.where(distances < threshold)[0]

    # Create a new random folder name for similar faces
    similar_faces_folder = os.path.join(face_recognition_folder, generate_random_name())
    if not os.path.exists(similar_faces_folder):
        os.makedirs(similar_faces_folder)

    # Copy similar images to the new folder
    for idx in similar_faces_indices:
        src_path = os.path.join(image_folder, comparison_labels[idx])
        dst_path = os.path.join(similar_faces_folder, comparison_labels[idx])
        copyfile(src_path, dst_path)
        print(f"Copied {src_path} to {dst_path}")

    # Also copy the target image to the new folder
    target_dst_path = os.path.join(similar_faces_folder, target_label)
    copyfile(os.path.join(image_folder, target_label), target_dst_path)
    print(f"Copied target image {os.path.join(image_folder, target_label)} to {target_dst_path}")


if __name__ == '__main__':
    image_folder = r"C:\Users\Daimon\Desktop\3500project\Raw_data"
    output_folder = r"C:\Users\Daimon\Desktop\3500project\data\album"

    print(f"Checking if the image folder exists: {os.path.exists(image_folder)}")
    print(f"Checking if the output folder exists: {os.path.exists(output_folder)}")

    find_similar_faces(image_folder, output_folder)
