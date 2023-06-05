from keras import regularizers
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from keras.optimizers import RMSprop
import keras



class CharacterRecognizer:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()

        model.add(Conv2D(input_shape=self.input_shape, activation='relu', filters=128, kernel_size=3, padding='same'))
        model.add(Conv2D(activation='relu', filters=32, kernel_size=5, padding='same'))
        model.add(Conv2D(activation='relu', filters=112, kernel_size=3, padding='same'))
        model.add(Dropout(0.4))

        model.add(Conv2D(activation='relu', filters=128, kernel_size=5, padding='same', strides=2))
        model.add(Conv2D(activation='relu', filters=128, kernel_size=3, padding='same', strides=3))

        model.add(Conv2D(activation='relu', filters=112, kernel_size=5, padding='same'))

        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(units=128, activation='relu'))
        model.add(Dropout(0.4))

        model.add(Dense(units=self.num_classes, activation='softmax'))

        return model

    def compile(self):
        self.model.compile(
            loss="categorical_crossentropy",
            # optimizer="adam",
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),

            metrics=["accuracy"]
        )

    def fit(self, X_train, y_train, X_test, y_test, epochs, batch_size=128):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        # self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), callbacks=[early_stopping])

        # для аугментированных данных
        self.model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                       validation_data=(X_test, y_test),
                       callbacks=[early_stopping],
                       epochs=epochs)

    def save(self, path):
        self.model.save(path)

if __name__ == '__main__':

    pass

    # recognizer = CharacterRecognizer(input_shape=(28, 28, 1), num_classes=y_train.shape[1])
    # recognizer.compile()
    # recognizer.fit(X_train, y_train, X_test, y_test, epochs=15)
    # # recognizer.save('character_recognition_model2.h5')
    # recognizer.save('best_model.h5')
