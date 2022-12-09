import argparse
import time

import carla
import cv2
import numpy as np

import model

IM_WIDTH = 200
IM_HEIGHT = 66


class RunCARLA():

    def __init__(self, driving_model: model.PilotNet, carla_ip: str, carla_port: int) -> None:
        world = self.setup_world(carla_ip, carla_port)
        self.driving_model = driving_model

        self.setup_vehicle(world)
        self.setup_camera(world)
        self.run(world)

    def setup_world(self, carla_ip: str, carla_port: int):
        client = carla.Client(carla_ip, carla_port)
        client.set_timeout(2.0)
        world = client.get_world()
        world.apply_settings(carla.WorldSettings(
            synchronous_mode=True,
            fixed_delta_seconds=0.05
        ))

        return world

    def setup_vehicle(self, world):
        print("Setting up Vehicle")
        # Setup Vehicle
        vehicle_blueprints = world.get_blueprint_library().filter("*vehicle*")
        spawn_points = world.get_map().get_spawn_points()
        self.vehicle = world.spawn_actor(np.random.choice(vehicle_blueprints), np.random.choice(spawn_points))
        # Give some time to CARLA to initialize
        time.sleep(5)
        print("Done")

    def setup_camera(self, world):
        print("Setting up camera")
        camera_init_trans = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera_blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
        camera_blueprint.set_attribute("image_size_x", str(IM_WIDTH))
        camera_blueprint.set_attribute("image_size_y", str(IM_HEIGHT))
        # Set Field of View of camera
        camera_blueprint.set_attribute("fov", "110")

        self.camera = world.spawn_actor(camera_blueprint, camera_init_trans, attach_to=self.vehicle)
        self.camera.listen(lambda image: self.process_image(image))
        print("Done")

    def process_image(self, image):
        try:
            image = np.array(image.raw_data)
            image = image.reshape((IM_HEIGHT, IM_WIDTH, 4))
            image = image[:, :, :3]

            # steering_angle, throttle_press, brake_pressure = self.driving_model.predict(image.reshape(-1, IM_HEIGHT, IM_WIDTH, 3))
            steering_angle = self.driving_model.predict(image.reshape(-1, IM_HEIGHT, IM_WIDTH, 3))

            # print(f"Steering: {steering_angle}, Throttle: {throttle_press}, Brake: {brake_pressure}")
            print(f"Steering: {steering_angle[0][0]}")
            control = carla.VehicleControl(throttle=0.4, steer=float(steering_angle[0][0]))
            self.vehicle.apply_control(control)

            cv2.imshow("Display", image)
            cv2.waitKey(1)
        except KeyboardInterrupt:
            print("exiting from process image")
            self.camera.stop()
            self.vehicle.destroy()

    def run(self, world):
        print("Running")
        while True:
            try:
                world.tick()
            except KeyboardInterrupt:
                print("Destroying vehicle")
                self.camera.stop()
                self.vehicle.destroy()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="Run PilotNet in CARLA")
    parser.add_argument("model_path", type=str)
    parser.add_argument("--carla-ip", type=str, default="localhost")
    parser.add_argument("--carla-port", type=int, default=2000)
    args = parser.parse_args()

    driving_model = model.PilotNet(IM_WIDTH, IM_HEIGHT, path_to_model=args.model_path)
    run_carla = RunCARLA(driving_model, args.carla_ip, args.carla_port)
