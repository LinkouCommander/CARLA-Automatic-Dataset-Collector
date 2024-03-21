import glob
import os
import sys
'''
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
'''
import carla
import math
import random
from queue import Queue
from queue import Empty
import numpy as np

output_path = '../project/image'

image_count = 0

# save image
def sensor_callback(sensor_data, sensor_queue, sensor_name):
    global image_count
    if 'camera' in sensor_name:
        if image_count % 10 == 0:
            sensor_data.save_to_disk(os.path.join(output_path, '%06d.png' % sensor_data.frame))
        sensor_queue.put((sensor_data.frame, sensor_name))
        image_count += 1

def main():
    count = 0
    actor_list = []
    sensor_list = []

    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world  = client.get_world()
    original_settings = world.get_settings()

    try:

        bp_lib = world.get_blueprint_library()

        # Set up the simulator in synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True # Enables synchronous mode
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

        sensor_queue = Queue()

        # spawn vehicle
        vehicle_bp =bp_lib.find('vehicle.lincoln.mkz_2020')
        # Get the map spawn points
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
        # Set up automatic drive
        vehicle.set_autopilot(True)
        # collect all actors to destroy when we quit the script
        actor_list.append(vehicle)

        # spawn camera
        camera_bp = bp_lib.find('sensor.camera.rgb')
        # Set camera blueprint properties
        camera_bp.set_attribute('bloom_intensity','1')
        camera_bp.set_attribute('fov','100')
        camera_bp.set_attribute('slope','0.7')
        # camera_bp.set_attribute('shutter_speed','0.00005')
        # camera_bp.set_attribute('sensor_tick','0.1')
        # camera position related to the vehicle
        camera_init_trans = carla.Transform(carla.Location(x=1.5, z=1.5))
        camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
        # output_path = os.path.join("../project/image", '%06d.png')
        
        if not os.path.exists(output_path): 
            os.makedirs(output_path)
        camera.listen(lambda image: sensor_callback(image, sensor_queue, "camera"))
        # Create a queue to store and retrieve the sensor data
        sensor_list.append(camera)


        while True:
            world.tick()
            '''
            # set the sectator to follow the ego vehicle
            spectator = world.get_spectator()
            transform = vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20), carla.Rotation(pitch=-90)))
            '''
            world.get_spectator().set_transform(camera.get_transform())

            try:
                for i in range(0, len(sensor_list)):
                    s_frame = sensor_queue.get(True, 1.0)
                    print("    Frame: %d   Sensor: %s" % (s_frame[0], s_frame[1]))

            except Empty:
                print("   Some of the sensor information is missed")


    finally:
        world.apply_settings(original_settings)
        print('destroying actors')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        for sensor in sensor_list:
            sensor.destroy()
        print('done.')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')