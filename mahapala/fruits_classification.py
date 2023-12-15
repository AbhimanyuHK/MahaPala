"""

Created by abhimanyu at 16/06/21

"""

# importing libraries

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class FruitsClassification:

    def __init__(self, x_path: str = ""):
        self.history = None
        self.class_names = []
        self.img_width, self.img_height = 224, 224

        self.train_data_dir = x_path + 'data/fruits/train'
        self.validation_data_dir = x_path + 'data/fruits/test'
        self.predict_data_dir = x_path + 'data/fruits/predict'

        self.nb_train_samples = 400
        self.nb_validation_samples = 100
        self.epochs = 10
        self.batch_size = 1
        self.num_classes = 5

        self.model = Sequential([
            layers.experimental.preprocessing.Rescaling(
                1. / 255,
                input_shape=(self.img_height, self.img_width, 3)
            ),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.num_classes)
        ])

        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=['accuracy']
        )
        # print(model.summary())

    def train_data(self):
        train_generator = tf.keras.preprocessing.image_dataset_from_directory(
            self.train_data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )
        self.class_names = train_generator.class_names
        print(self.class_names)
        return train_generator

    def validation_data(self):
        validation_generator = tf.keras.preprocessing.image_dataset_from_directory(
            self.validation_data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )

        return validation_generator

    def fit(self):
        self.history = self.model.fit(
            self.train_data(),
            validation_data=self.validation_data(),
            epochs=self.epochs
        )

    def save_model(self):
        self.model.save_weights('model_saved.h5')

    def analysis(self):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def predict(self, prediction_path=None):
        lt = []
        predict_generator = tf.keras.preprocessing.image_dataset_from_directory(
            prediction_path if prediction_path else self.predict_data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )
        print(predict_generator)

        # image = tf.keras.utils.load_img("../data/fruits/predict/test/21.jpeg")
        # print("image", image)
        # input_arr = keras.utils.img_to_array(image)
        # predict_generator = np.array([input_arr])  # Convert single image to a batch.

        f = self.model.predict(predict_generator)
        score = tf.nn.softmax(f[0])
        data = {
            "class": self.class_names[np.argmax(score)],
            "confidence": "{:.2f} percent".format(100 * np.max(score))
        }
        print(data)
        lt.append(data)
        return lt

    def predict_image(self, image):
        self.model.summary()
        lt = []

        predict_generator = tf.keras.preprocessing.image.img_to_array(
            image
        )
        predict_generator = tf.expand_dims(predict_generator, axis=0)  # add an extra dimension at axis 0
        print(predict_generator.shape)

        f = self.model.predict([predict_generator])
        score = tf.nn.softmax(f[0])
        data = {
            "class": self.class_names[np.argmax(score)],
            "confidence": "{:.2f} percent".format(100 * np.max(score))
        }
        print(data)
        lt.append(data)
        return lt


if __name__ == "__main__":
    fc = FruitsClassification()
    fc.fit()
    r = fc.predict()
    print(r)
    # for x in range(10):
    #     r = fc.predict()
    #     print(r)
    #     input("Enter to continue")
