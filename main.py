import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the trained model
model = load_model('model/model.h5')

# Load the class labels
class_labels = ['normal', 'pothole']

# Function to preprocess the input image


def preprocess_image(image_path):
    image = load_img(image_path, target_size=(128, 128))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict the class of the image


def predict_class(image):
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    class_label = class_labels[class_index]
    return class_label


# Get the input image path from the user
image_path = input("Enter the path to the image: ")

# Preprocess the image
image = preprocess_image(image_path)

# Loop to process images continuously
while True:
    # Predict the class of the image
    predicted_class = predict_class(image)

    # Load the image in color mode
    image_display = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Fix resize on image_display
    image_display = cv2.resize(image_display, (700, 700))

    # Add the predicted class as text in the top-left corner
    text = predicted_class
    cv2.putText(image_display, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Display the image
    cv2.imshow('Prediction', image_display)

    # Wait for the key press
    key = cv2.waitKey(1)
    if key == 27:  # Esc key
        break

# Close the window
cv2.destroyAllWindows()
