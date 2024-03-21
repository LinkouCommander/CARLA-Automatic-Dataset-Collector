import ast
import csv
import pandas as pd

# code you want to digest
code = """

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

def sensor_callback(sensor_data, sensor_queue, sensor_name):
    if 'camera' in sensor_name:
        sensor_data.save_to_disk(os.path.join('../project/image', '%06d.png' % sensor_data.frame))
    sensor_queue.put((sensor_data.frame, sensor_name))

def main():
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
        camera_bp.set_attribute('sensor_tick','0.1')
        # camera position related to the vehicle
        camera_init_trans = carla.Transform(carla.Location(x=1.5, z=1.5))
        camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
        # Create a queue to store and retrieve the sensor data
        # camera.listen(image_queue.put)
        # output_path = os.path.join("../project/image", '%06d.png')
        output_path = '../project/image'
        if not os.path.exists(output_path): 
            os.makedirs(output_path)
        camera.listen(lambda image: sensor_callback(image, sensor_queue, "camera"))
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
"""

# transform AST into CSV format
def ast_to_csv(node, csv_writer, parent=None):
     node_str = ast.dump(node)
     if parent != None:
        csv_writer.writerow([parent, node_str])  # 在每行中包括父節點和子節點之間的關係
     for child_node in ast.iter_child_nodes(node):
         ast_to_csv(child_node, csv_writer, node_str)  # 傳遞當前節點的字串表示作為下一個節點的父節點 

with open('ast_tree.csv', 'w', newline='') as csvfile:
     csv_writer = csv.writer(csvfile)
     # csv_writer.writerow(['Source', 'Target'])
     ast_tree = ast.parse(code)
     ast_to_csv(ast_tree, csv_writer)

# def ast_to_csv(node, csv_writer, node_id):
#     node_str = ast.dump(node)
#     csv_writer.writerow([node_id, node_str])  # 在每行中包括父節點和子節點之間的關係
#     for child_node in ast.iter_child_nodes(node):
#         ast_to_csv(child_node, csv_writer, node_id + 1)  # 傳遞當前節點的字串表示作為下一個節點的父節點

# with open('ast_ID.csv', 'w', newline='') as csvfile:
#     csv_writer = csv.writer(csvfile)
#     csv_writer.writerow(['Id', 'Label'])
#     ast_tree = ast.parse(code)
#     ast_to_csv(ast_tree, csv_writer, 1)  # 從1開始編號


# Read the CSV file
df = pd.read_csv('ast_tree.csv', names=['Source', 'Target'])

# Create a new DataFrame to store unique labels
df1 = pd.DataFrame(columns=['Id', 'Label'])
i = 1

# Iterate over the 'Source' and 'Target' columns
for index, row in df.iterrows():
    # Check if the value in the 'Source' column exists in df1
    if row['Source'] not in df1['Id'].values:
        df1.loc[len(df1)] = [row['Source'], i]
        # df.at[index, 'Source'] = i
        i += 1
    # else:
        # Update the 'Source' value with its corresponding index in df1
        # df.at[index, 'Source'] = df1.loc[df1['Label'] == row['Source'], 'Id'].iloc[0]

    # Check if the value in the 'Target' column exists in df1
    if row['Target'] not in df1['Id'].values:
        df1.loc[len(df1)] = [row['Target'], i]
        # df.at[index, 'Target'] = i
        i += 1
    # else:
        # Update the 'Target' value with its corresponding index in df1
        # df.at[index, 'Target'] = df1.loc[df1['Label'] == row['Target'], 'Id'].iloc[0]

df1.to_csv('ast_nodes.csv', index=False)