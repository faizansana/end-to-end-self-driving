import json
import os
from typing import Tuple

import cv2
import sklearn.model_selection


class ImageData(object):

    def __init__(self, image_path: str, image_filename: str, image_shape: Tuple[int, int] = (160, 120), training_set=True) -> None:
        if training_set:
            self.steering, self.throttle, self.brake, self.image = self.parse_train(image_path, image_filename, image_shape)
        else:
            self.steering, self.throttle, self.brake, self.image = self.parse_test(image_path, image_shape)

    def parse_train(self, image_path: str, image_filename: str, image_shape: Tuple[int, int] = (160, 120)):
        data = json.loads(image_filename[:-4])

        image = cv2.imread(f"{image_path}{image_filename}")
        image = cv2.resize(image, image_shape)
        return (data[1], data[2], data[3], image)

    def parse_test(self, image_path: str, image_shape: Tuple[int, int] = (160, 120)):

        image = cv2.imread(filename=image_path)
        image = cv2.resize(image, image_shape)
        image = image.reshape(1, image_shape[1], image_shape[0], 3)
        return (0, 0, 0, image)


class Data(object):

    def __init__(self, image_shape: Tuple[int, int] = (160, 120)) -> None:
        self.data = self.generate_data(image_shape)
        self.training_data, self.test_data = sklearn.model_selection.train_test_split(self.data, test_size=0.2, shuffle=False)

    def generate_data(self, image_shape: Tuple[int, int] = (160, 120)):
        data = []
        with os.scandir("recordings/") as recordings:
            for recording in recordings:
                with os.scandir(recording) as images:
                    for image in images:
                        data.append(ImageData(f"./recordings/{recording.name}/", image.name, image_shape))

        return data


if __name__ == "__main__":
    test = Data()