from ocr_module.preprocessing import ImageProcessor
from ocr_module.text_recognition import TextRecognizer
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def get_text_from_image(model_name,image):

    model = load_model(model_name)
    processor = ImageProcessor()
    recognizer = TextRecognizer(model, processor)


    res = recognizer.recognize_text(image)

    # print(res)
    return res

if __name__ == '__main__':

    # model_name = 'models/character_recognition_model2.h5'
    model_name = 'models/ua_model.h5'
    image_path = 'images/noise.png'

    image = cv2.imread(image_path)

    res = get_text_from_image(model_name, image)
    print(res)