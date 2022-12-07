import datetime
import os
from typing import Tuple

import carla
import cv2
import numpy as np


class DataGenerator(object):

    def __init__(self, world, time: int, img_width: int = 640, img_height: int = 480) -> None:
        self.start_time = datetime.datetime.now()
        self.world = world
        self.img_width = img_width
        self.img_height = img_height
        self.vehicle = None
        self.interrupt = False

        self.directory = f"recordings/{datetime.datetime.now().strftime('%Y-%m-%d@%H.%M.%S' if os.name is 'nt' else '%Y-%m-%d@%H:%M:%S')}"
        self.start(time)

    def record(self, image):
        control = self.vehicle.get_control()
        image.save_to_disk(f"{self.directory}/{[int((datetime.datetime.now() - self.start_time).total_seconds()), control.steer, control.throttle, control.brake]}.png")

        display_image = np.array(image.raw_data)
        # Convert to RGBA
        display_image = display_image.reshape((self.img_height, self.img_width, 4))
        # Convert from RGBA to RGB
        display_image = display_image[:, :, : 3]

        cv2.imshow(winname="Display", mat=display_image)

        if cv2.waitKey(5) == 27:
            self.interrupt = True

    def start(self, time: int):

        # Make all traffic lights green
        traffic_lights = world.get_actors().filter('traffic.traffic_light')
        for traffic_light in traffic_lights:
            traffic_light.set_state(carla.TrafficLightState.Green)
            traffic_light.freeze(True)

        # get a list of spawn points & vehicles then randomly choose from one to use
        print("Spawning vehicle")
        vehicle_blueprints = self.world.get_blueprint_library().filter("*vehicle*")
        spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle = self.world.spawn_actor(np.random.choice(vehicle_blueprints), np.random.choice(spawn_points))
        print('OK')

        print('Spawning camera and attaching to vehicle')
        camera_init_trans = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera_blueprint = self.world.get_blueprint_library().find("sensor.camera.rgb")
        camera_blueprint.set_attribute("image_size_x", str(self.img_width))
        camera_blueprint.set_attribute("image_size_y", str(self.img_height))
        camera_blueprint.set_attribute("fov", "110")  # sets field of view (FOV)
        print("OK")

        # Attach camera to vehicle and start recording
        self.camera = self.world.spawn_actor(camera_blueprint, camera_init_trans, attach_to=self.vehicle)
        self.camera.listen(lambda image: self.record(image))

        # autopilot obviously
        self.vehicle.set_autopilot(True)

        try:

            while True:
                if self.interrupt:
                    self.stop()
                    break

                self.world.tick()
        except KeyboardInterrupt:
            print("Keyboard Exit!")
            self.stop()

    def stop(self):
        print("Exiting Recorder")

        self.camera.stop()
        self.vehicle.destroy()


if __name__ == "__main__":
    client = carla.Client("localhost", 2000)
    world = client.get_world()
    collector = DataGenerator(world, 10)
