from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

from ocr_module.models.model import CharacterRecognizer


df = pd.read_csv("dataset/processed_uaset.csv")

X = df[df.columns[1:]]
y = df['label']

# Нормалізація даних
X = X / 255.0

X = 1 - X

#  Розділяємо дані на навчальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Змінюємо форму даних, щоб вони підходили для CNN
X_train = X_train.values.reshape(-1, 28, 28, 1)
X_test = X_test.values.reshape(-1, 28, 28, 1)

# Перетворюємо мітки в one-hot вектори
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


class ImageDataGeneratorWithNoise(ImageDataGenerator):
    def __init__(self, noise_stddev=0.1, **kwargs):
        super().__init__(**kwargs)
        self.noise_stddev = noise_stddev

    def random_transform(self, x):
        x = super().random_transform(x)
        noise = np.random.normal(scale=self.noise_stddev, size=x.shape)
        return np.clip(x + noise * 0.2, 0., 1.)


# новий генератор з шумом
datagen = ImageDataGeneratorWithNoise(
    zoom_range=0.1,
    height_shift_range=0.1,
    noise_stddev=0.1
)

datagen.fit(X_train)

recognizer = CharacterRecognizer(input_shape=(28, 28, 1), num_classes=y_train.shape[1])
recognizer.compile()
recognizer.fit(X_train, y_train, X_test, y_test, epochs=15)
recognizer.save('best_model.h5')
