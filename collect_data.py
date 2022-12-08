import datetime
import os

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
        self.time_elapsed = None
        # self.interrupt = False
        self.start_time = datetime.datetime.now()
        self.required_time = datetime.timedelta(minutes=time)
        # Make world synchronous
        self.setup_sim_world(world)

        self.directory = f"recordings/{datetime.datetime.now().strftime('%Y-%m-%d@%H.%M.%S' if os.name == 'nt' else '%Y-%m-%d@%H:%M:%S')}"
        self.start(self.required_time)

    def setup_sim_world(self, world):
        world.apply_settings(carla.WorldSettings(
            synchronous_mode=True,
            fixed_delta_seconds=0.05
        ))

    def record(self, image, display_image=False):
        try:
            control = self.vehicle.get_control()
        except RuntimeError:
            print("Getting new actor")
            self.start(self.required_time - self.time_elapsed)
        image.save_to_disk(f"{self.directory}/{[int((datetime.datetime.now() - self.start_time).total_seconds()), control.steer, control.throttle, control.brake]}.png")

        if display_image:
            self.display_image(image)

    def display_image(self, image):
        display_image = np.array(image.raw_data)
        # Convert to RGBA
        display_image = display_image.reshape((self.img_height, self.img_width, 4))
        # Convert from RGBA to RGB
        display_image = display_image[:, :, : 3]

        cv2.imshow(winname="Display", mat=display_image)

        # if cv2.waitKey(5) == 27:
        #     self.interrupt = True
        cv2.waitKey(2)

    def spawn_vehicle(self):
        # get a list of spawn points & vehicles then randomly choose from one to use
        vehicle_blueprints = self.world.get_blueprint_library().filter("*vehicle*")
        spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle = self.world.spawn_actor(np.random.choice(vehicle_blueprints), np.random.choice(spawn_points))

    def spawn_camera(self):
        camera_init_trans = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera_blueprint = self.world.get_blueprint_library().find("sensor.camera.rgb")
        camera_blueprint.set_attribute("image_size_x", str(self.img_width))
        camera_blueprint.set_attribute("image_size_y", str(self.img_height))
        camera_blueprint.set_attribute("fov", "110")  # sets field of view (FOV)

        # Attach camera to vehicle and start recording
        self.camera = self.world.spawn_actor(camera_blueprint, camera_init_trans, attach_to=self.vehicle)
        self.camera.listen(lambda image: self.record(image))

        # Set Vehicle in Autopilot Mode
        self.vehicle.set_autopilot(True)

    def start(self, required_time: datetime.timedelta):

        # Make all traffic lights green
        traffic_lights = world.get_actors().filter('traffic.traffic_light')
        for traffic_light in traffic_lights:
            traffic_light.set_state(carla.TrafficLightState.Green)
            traffic_light.freeze(True)

        print("Spawning vehicle")
        self.spawn_vehicle()
        print('OK')

        print('Spawning camera and attaching to vehicle')
        self.spawn_camera()
        print("OK")

        try:
            start_time = datetime.datetime.now()
            time_elapsed = datetime.datetime.now() - start_time

            while True:
                if time_elapsed >= required_time:
                    self.stop()
                    break

                time_elapsed = datetime.datetime.now() - start_time
                total_time_elapsed = datetime.datetime.now() - self.start_time
                # To call it from reward function
                self.time_elapsed = time_elapsed
                print("Elapsed time:", time_elapsed.seconds)
                print("Total Elapsed time:", total_time_elapsed.seconds)
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
    collector = DataGenerator(world, 30)
