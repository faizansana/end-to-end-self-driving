import numpy as np
import tensorflow as tf


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

        # Convolutional layers
        x = tf.keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation="relu")(inputs)
        x = tf.keras.layers.Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation="relu")(x)
        x = tf.keras.layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation="relu")(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu")(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu")(x)

        # Flatten
        x = tf.keras.layers.Flatten()(x)

        # Dense Layers
        x = tf.keras.layers.Dense(units=100, activation="relu")(x)
        x = tf.keras.layers.Dense(units=50, activation="relu")(x)
        x = tf.keras.layers.Dense(units=10, activation="relu")(x)

        # Modified
        steering_angle = tf.keras.layers.Dense(units=1)(x)
        steering_angle = tf.keras.layers.Lambda(lambda X: tf.multiply(tf.atan(X), 2), name="steering_angle")(steering_angle)

        throttle_press = tf.keras.layers.Dense(units=1)(x)
        throttle_press = tf.keras.layers.Lambda(lambda X: tf.multiply(tf.atan(X), 2), name="throttle_press")(throttle_press)

        brake_pressure = tf.keras.layers.Dense(units=1, activation="linear")(x)
        brake_pressure = tf.keras.layers.Lambda(lambda X: tf.multiply(tf.atan(X), 2), name="brake_pressure")(brake_pressure)

        model = tf.keras.Model(inputs=[inputs], outputs=[steering_angle, throttle_press, brake_pressure])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss={"steering_angle": "mse", "throttle_press": "mse", "brake_pressure": "mse"}
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

    def predict(self, data: np.ndarray):
        return self.model.predict(data)

    def train(self, model_name: str, data, epochs: int = 30, steps_per_epoch: int = 10, steps_val: int = 10, batch_size: int = 64):
        # self.model.fit()
        pass


if __name__ == "__main__":
    model = PilotNet(66, 200)
