from ocr_module.preprocessing import ImageProcessor
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


class TextRecognizer:
    def __init__(self, model, image_processor):
        self.model = model
        self.image_processor = image_processor
        self.ukr_alphabet = "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯабвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
        # self.ukr_alphabet = 'АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯабвгґдеєжзиіїйклмнопрстуфхцчшщьюя0987654321ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!"#$%&’()*+,-./:;<=>?@[]^_`{|}~'
        # self.ukr_alphabet = 'АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯабвгґдеєжзиіїйклмнопрстуфхцчшщьюя0987654321!"#%’()*+,-./:;'
        self.labels_to_letters = {i: self.ukr_alphabet[i] for i in range(len(self.ukr_alphabet))}

    def letter_to_class(self, letter):
        return self.ukr_alphabet.index(letter)

    def class_to_letter(self, cls):
        return self.labels_to_letters[cls]

    def crop_img(self, img):
        if img is None or img.size == 0:
            raise ValueError("Invalid input image")

        coords = np.column_stack(np.where(img > 0))

        if coords.size == 0:
            raise ValueError("No non-zero pixels found")

        # обчислюємо обмежувальний прямокутник для ненульових координат і обрізаємо зображення
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        cropped_img = img[x_min:x_max + 1, y_min:y_max + 1]

        if cropped_img.size == 0:
            raise ValueError("Cropped image is empty")

        # Обчислюємо нові розміри зображення, зберігаючи співвідношення сторін
        height, width = cropped_img.shape[:2]
        if height > width:
            new_height = 28
            new_width = int(width * new_height / height)
        else:
            new_width = 28
            new_height = int(height * new_width / width)

        # print(cropped_img.shape)
        # print(new_width)
        # print(new_height)
        # print('----------')

        resized_img = cv2.resize(cropped_img, (new_width, new_height))

        if resized_img.size == 0:
            raise ValueError("Resized image is empty")

        # ширина і висота білих смуг, які потрібно додати
        padding_height = 28 - new_height
        padding_width = 28 - new_width

        if padding_height < 0 or padding_width < 0:
            raise ValueError("Padding is negative. Check the resized image dimensions.")

        top_border_height = padding_height // 2
        bottom_border_height = padding_height - top_border_height

        left_border_width = padding_width // 2
        right_border_width = padding_width - left_border_width

        padded_img = cv2.copyMakeBorder(resized_img, top_border_height, bottom_border_height, left_border_width,
                                        right_border_width, cv2.BORDER_CONSTANT, value=255)

        padded_img = cv2.bitwise_not(padded_img)

        return padded_img

    def preprocess_image(self, image):
        image = cv2.resize(image, (28, 28))

        if len(image.shape) > 2 and image.shape[2] > 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = np.reshape(image, (28, 28, 1))

        image = image / 255.0
        return image

    def predict_letter(self, image):
        # plt.imshow(image)
        # plt.show()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = self.crop_img(image)
        # image = 255 - image
        image = self.preprocess_image(image)

        # TODO обробити попередньо зображення через функцію crop_img
        # TODO мабуть винести цю частину с функції

        # reshaped_img = image.reshape(28, 28)
        # image = self.preprocess_image(image)

        # ----------------
        prediction = self.model.predict(np.array([image]))
        predicted_class = np.argmax(prediction)
        predicted_letter = self.class_to_letter(predicted_class)

        return predicted_letter

    def predict_letters(self, word_image):
        letters_images_resized = []
        letters_positions = self.image_processor.get_letters_positions(word_image)

        for pos in letters_positions:
            x, y, w, h = pos
            letter_img = word_image[y:y + h, x:x + w]

            # TODO обробити попередньо зображення через функцію crop_img
            # TODO мабуть винести цю частину с функції
            # TODO чи потрібна тоді функція predict_letter?

            resized_img = cv2.resize(letter_img, (28, 28))  # Приведение к одному размеру
            gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            letters_images_resized.append(gray_img[..., np.newaxis])  # Добавляем измерение для каналов

        if letters_images_resized:
            letters_images_array = np.stack(letters_images_resized, axis=0)
        else:
            return ''

        predictions = self.model.predict(letters_images_array)

        predicted_classes = [np.argmax(p) for p in predictions]
        predicted_letters = [self.class_to_letter(c) for c in predicted_classes]

        return predicted_letters

    def recognize_text(self, image):

        self.image_processor.gray = image
        # self.image_processor.gray = self.image_processor.to_gray(self.image_processor.img)
        self.image_processor.gray = self.image_processor.contrast(self.image_processor.gray)
        self.image_processor.gray = self.image_processor.as_white_background(self.image_processor.gray)
        lines = self.image_processor.get_lines_positions(self.image_processor.gray)

        text = ""
        for line in lines:
            start, end = line
            line_image = self.image_processor.gray[start:end]
            # plt.imshow(line_image)
            # plt.show()

            words = self.image_processor.get_words_positions(line_image)
            # print(words)
            # plt.imshow(self.image_processor.gray)
            # plt.show()

            # plt.imshow(line_image)
            # plt.show()
            # i = 0
            for word in words:
                # i+=1

                word_image = self.image_processor.crop_word(line_image, word)
                # cv2.imwrite(f'/content/word_{i}.jpg', word_image)

                letter_predictions = self.predict_letters(word_image)
                word_text = "".join(letter_predictions)
                text += word_text + " "  # Add a space after each word
            text += "\n"  # Add a newline after each line
        return text


if __name__ == '__main__':


    # model = load_model('models/character_recognition_model2.h5')
    model = load_model('models/ua_model.h5')
    processor = ImageProcessor()
    recognizer = TextRecognizer(model, processor)

    image = cv2.imread('images/noise.png')

    res = recognizer.recognize_text(image)

    print(res)