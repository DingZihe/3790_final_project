# coding:utf-8
import os
from numpy import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']


# Convert image to vector
def image_to_vector(image_path):
    image = cv2.imread(image_path, 0)  # Read image
    num_rows, num_cols = image.shape
    vectorized_image = np.zeros((1, num_rows * num_cols))
    vectorized_image = np.reshape(image, (1, num_rows * num_cols))
    return vectorized_image

database_path = "ORL"


# Load face database, randomly select k images as training set for each person, the rest constitute test set
def load_face_database(num_train_images):

    training_data = np.zeros((40 * num_train_images, 112 * 92))
    training_labels = np.zeros(40 * num_train_images)  # [0,0,.....0](40*num_train_images zeros)
    testing_data = np.zeros((40 * (10 - num_train_images), 112 * 92))
    testing_labels = np.zeros(40 * (10 - num_train_images))
    shuffled_samples = random.permutation(10) + 1  # Randomly sort 1-10 (0-9) +1
    for person_idx in range(40):  # Total of 40 people
        person_id = person_idx + 1
        for img_idx in range(10):  # Each person has 10 images
            image_path = database_path + '/s' + str(person_id) + '/' + str(shuffled_samples[img_idx]) + '.jpg'
            # Read and vectorize the image
            vectorized_image = image_to_vector(image_path)
            if img_idx < num_train_images:
                # Construct training set
                training_data[person_idx * num_train_images + img_idx, :] = vectorized_image
                training_labels[person_idx * num_train_images + img_idx] = person_id
            else:
                # Construct test set
                testing_data[person_idx * (10 - num_train_images) + (img_idx - num_train_images), :] = vectorized_image
                testing_labels[person_idx * (10 - num_train_images) + (img_idx - num_train_images)] = person_id

    return training_data, training_labels, testing_data, testing_labels


# Define PCA algorithm
def perform_pca(data, num_components):
    data = np.float32(np.mat(data))
    num_rows, num_cols = np.shape(data)
    mean_data = np.mean(data, 0)  # Mean along columns
    centered_data = data - np.tile(mean_data, (num_rows, 1))  # Subtract mean from all samples to get centered_data
    covariance_matrix = centered_data * centered_data.T  # Get the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)  # Obtain eigenvalues and eigenvectors of the covariance matrix
    principal_components = eigenvectors[:, 0:num_components]  # Take the first num_components eigenvectors along columns
    principal_components = centered_data.T * principal_components  # Transition small matrix eigenvectors to large matrix eigenvectors
    for i in range(num_components):
        principal_components[:, i] = principal_components[:, i] / np.linalg.norm(principal_components[:, i])  # Normalize the eigenvectors

    reduced_data = centered_data * principal_components
    return reduced_data, mean_data, principal_components


# Face recognition
def face_recognition():
    for num_components in range(10, 41, 10):  # Reduce to at most 40 dimensions, i.e., select the first 40 principal components (because when num_train_images=1, there are only 40 dimensions)
        print(f"Reducing to {num_components} dimensions")
        x_values = []
        y_values = []
        for num_train_images in range(1, 10):
            train_data, train_labels, test_data, test_labels = load_face_database(num_train_images)  # Get the dataset

            # Train using PCA algorithm
            reduced_train_data, mean_data, principal_components = perform_pca(train_data, num_components)
            num_train_samples = reduced_train_data.shape[0]  # Total number of training faces
            num_test_samples = test_data.shape[0]  # Total number of test faces
            centered_test_data = test_data - np.tile(mean_data, (num_test_samples, 1))
            reduced_test_data = centered_test_data * principal_components  # Get test face data in the feature vector
            reduced_test_data = np.array(reduced_test_data)  # Convert mat to array
            reduced_train_data = np.array(reduced_train_data)

            # Test accuracy
            correct_predictions = 0
            for i in range(num_test_samples):
                test_face = reduced_test_data[i, :]
                difference_matrix = reduced_train_data - np.tile(test_face, (num_train_samples, 1))  # Distance between training data and test face
                squared_diff_matrix = difference_matrix ** 2
                squared_distances = squared_diff_matrix.sum(axis=1)  # Sum along rows
                sorted_distance_indices = squared_distances.argsort()  # Sort the vector from small to large, using indexes, get a vector
                nearest_index = sorted_distance_indices[0]  # Index of the nearest distance
                if train_labels[nearest_index] == test_labels[i]:
                    correct_predictions += 1
                else:
                    pass

            accuracy = float(correct_predictions) / num_test_samples
            x_values.append(num_train_images)
            y_values.append(round(accuracy, 2))

            print(f'Number of training images per person: {num_train_images}, Accuracy: {accuracy * 100:.2f}%')

        # Plot
        if num_components == 10:
            y1_values = y_values
        if num_components == 20:
            y2_values = y_values
        if num_components == 30:
            y3_values = y_values
        if num_components == 40:
            y4_values = y_values

    # Comparison of accuracies at different dimensions
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y1_values, 'o-', label="Reduced to 10 Dimensions")
    plt.plot(x_values, y2_values, 's-', label="Reduced to 20 Dimensions")
    plt.plot(x_values, y3_values, '^-', label="Reduced to 30 Dimensions")
    plt.plot(x_values, y4_values, 'd-', label="Reduced to 40 Dimensions")

    for x, y in zip(x_values, y1_values):
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=9)
    for x, y in zip(x_values, y2_values):
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=9)
    for x, y in zip(x_values, y3_values):
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=9)
    for x, y in zip(x_values, y4_values):
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=9)

    plt.legend(loc='lower right')
    plt.title("Comparison of Recognition Accuracy at Different Dimensions", fontsize=14)
    plt.xlabel("Number of Training Images per Person", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    face_recognition()
