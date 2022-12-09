import argparse
import datetime

import numpy as np
import tensorflow as tf

import data


class PilotNet():

    def __init__(self, width: int, height: int, path_to_model: str = "") -> None:
        self.image_width = width
        self.image_height = height

        if path_to_model:
            self.model = self.load_model(path_to_model)
        else:
            self.model = self.build_model()

    def build_model(self):
        inputs = tf.keras.Input(shape=(self.image_height, self.image_width, 3))

        img_normalized = tf.keras.layers.BatchNormalization()(inputs)
        # Convolutional layers
        x = tf.keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation="relu")(img_normalized)
        x = tf.keras.layers.Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation="relu")(x)
        x = tf.keras.layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation="relu")(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu")(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu")(x)
        # Flatten
        x = tf.keras.layers.Flatten()(x)

        # Dense Layers
        x = tf.keras.layers.Dense(units=100, activation="relu")(x)
        x = tf.keras.layers.Dropout(rate=0.1)(x)
        x = tf.keras.layers.Dense(units=50, activation="relu")(x)
        x = tf.keras.layers.Dropout(rate=0.1)(x)
        x = tf.keras.layers.Dense(units=10, activation="relu")(x)

        # Modified
        steering_angle = tf.keras.layers.Dense(units=1)(x)
        # steering_angle = tf.keras.layers.Lambda(lambda X: tf.multiply(tf.atan(X), 2), name="steering_angle")(steering_angle)

        # throttle_press = tf.keras.layers.Dense(units=1)(x)
        # throttle_press = tf.keras.layers.Lambda(lambda X: tf.multiply(tf.atan(X), 2), name="throttle_press")(throttle_press)

        # brake_pressure = tf.keras.layers.Dense(units=1)(x)
        # brake_pressure = tf.keras.layers.Lambda(lambda X: tf.multiply(tf.atan(X), 2), name="brake_pressure")(brake_pressure)

        # model = tf.keras.Model(inputs=[inputs], outputs=[steering_angle, throttle_press, brake_pressure])
        model = tf.keras.Model(inputs=[inputs], outputs=[steering_angle])

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-4, decay_steps=10000, decay_rate=0.9)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            # loss={"steering_angle": "mse", "throttle_press": "mse", "brake_pressure": "mse"}
            loss="mse",
            metrics="mae"
        )
        model.summary()
        return model

    def load_model(self, path_to_model: str):
        try:
            model = tf.keras.models.load_model(path_to_model)
        except IOError:
            print("Failed to load model")
            SystemExit()

        return model

    def predict(self, data: np.ndarray, batch_size: int = 1):
        return self.model.predict(data, batch_size=batch_size)

    def train(self, model_name: str, data_class: data.Data, epochs: int = 30, steps_per_epoch: int = 10, steps_val: int = 10, batch_size: int = 64):
        # self.model.fit()
        x_train, y_train = data_class.get_training_data()
        x_test, y_test = data_class.get_test_data()

        # Setting tensorboard
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        # Setting checkpoint
        checkpoint = tf.keras.callbacks.ModelCheckpoint(f"./models/{model_name}", monitor="loss", verbose=1, save_best_only=True)

        self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_split=0.2,
            validation_steps=steps_val,
            callbacks=[tensorboard_callback, checkpoint])

        model_stats = self.model.evaluate(x_test, y_test)

        return model_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train PilotNet model")
    parser.add_argument("--image-width", type=int, default=200, help="Input image width")
    parser.add_argument("--image-height", type=int, default=66, help="Input image height")
    parser.add_argument("--model_name", type=str, default="new_trained_model", help="Name of new trained model")
    parser.add_argument("--model_path", type=str, default="", help="Path of model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    args = parser.parse_args()

    image_width = args.image_width
    image_height = args.image_height

    print("Loading data")
    data_class = data.Data((image_width, image_height))
    print("Finished loading data")
    model = PilotNet(image_width, image_height, path_to_model=args.model_path)

    if args.model_path == "":
        stats = model.train("fourth_test_1000_epochs_batch_norm", data_class, epochs=args.epochs)
