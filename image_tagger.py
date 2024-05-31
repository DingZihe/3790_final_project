import os
import numpy as np
import urllib.request
import pickle
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image


def load_classification_model():
    model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
    input_shape = (224, 224)
    model = tf.keras.Sequential([
        hub.KerasLayer(model_url, input_shape=input_shape + (3,))
    ])
    return model


def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)


def predict_image_label(image_path, model):
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)

    labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
    with urllib.request.urlopen(labels_url) as response:
        labels = np.array(response.read().decode('utf-8').splitlines())

    predicted_label = labels[np.argmax(predictions)]
    return predicted_label.lower()


def generate_label_dictionary(data_directory):
    label_dictionary = {}

    model = load_classification_model()
    for person_folder in os.listdir(data_directory):
        person_folder_path = os.path.join(data_directory, person_folder)
        if os.path.isdir(person_folder_path):
            for filename in os.listdir(person_folder_path):
                if filename.endswith(".jpg"):
                    image_path = os.path.join(person_folder_path, filename)
                    predicted_label = predict_image_label(image_path, model)

                    if predicted_label in label_dictionary:
                        label_dictionary[predicted_label].append(image_path)
                    else:
                        label_dictionary[predicted_label] = [image_path]

    with open("label_dictionary.pickle", "wb") as f:
        pickle.dump(label_dictionary, f)


if __name__ == "__main__":
    data_directory = "data/album"
    generate_label_dictionary(data_directory)
