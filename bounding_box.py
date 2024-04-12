import glob
import os
import sys
import cv2
import carla
import math
import random
from queue import Queue
from queue import Empty
import numpy as np
# from pascal_voc_writer import Writer
import bbox_filter as cva

output_path = '../project/image'

# project 3D point to 2D
def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

# use the camera projection matrix to project the 3D points in camera coordinates into the 2D camera plane
def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]

def draw_boundingbox(img, x_min, y_min, x_max, y_max, color=(0,0,255,255)):
    cv2.line(img, (int(x_min),int(y_min)), (int(x_max),int(y_min)), color, 1)
    cv2.line(img, (int(x_min),int(y_max)), (int(x_max),int(y_max)), color, 1)
    cv2.line(img, (int(x_min),int(y_min)), (int(x_min),int(y_max)), color, 1) 
    cv2.line(img, (int(x_max),int(y_min)), (int(x_max),int(y_max)), color, 1)

def main():
    vehicle_num = 100
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
        settings.fixed_delta_seconds = 0.03
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

        # spawn vehicle
        vehicle_bp =bp_lib.find('vehicle.lincoln.mkz_2020')
        # Get the map spawn points
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
        # Set up automatic drive
        vehicle.set_autopilot(True)
        # collect all actors to destroy when we quit the script
        actor_list.append(vehicle)

        # generate npc vehicle
        for i in range(vehicle_num):
            # vehicle_npc = random.choice(bp_lib.filter('vehicle'))
            vehicle_npc = bp_lib.find('vehicle.carlamotors.firetruck')
            npc = world.try_spawn_actor(vehicle_npc, random.choice(spawn_points))

            # set light state
            # light_state = carla.VehicleLightState(carla.VehicleLightState.Special1 | carla.VehicleLightState.Special2)
            if npc:
                npc.set_autopilot(True)
                # apply light state
                # npc.set_light_state(light_state)
                actor_list.append(npc)

        # generate npc vehicle
        # for i in range(70):
        #     pedestrian_npc = random.choice(bp_lib.filter('pedestrian'))
        #     pedestrian_controller = world.get_blueprint_library().find('controller.ai.walker')

        #     npc = world.try_spawn_actor(pedestrian_npc, random.choice(spawn_points))
        #     # world.wait_for_tick()

        #     controller = world.spawn_actor(pedestrian_controller, carla.Transform(), npc)
        #     # world.wait_for_tick()
        #     actor_list.append(npc) 
        q_list = []
        idx = 0
        tick_queue = Queue()
        world.on_tick(tick_queue.put)
        q_list.append(tick_queue)
        tick_idx = idx
        idx = idx+1

        ### spawn camera
        camera_bp = bp_lib.find('sensor.camera.rgb')
        # Set camera blueprint properties
        camera_bp.set_attribute('bloom_intensity','1')
        camera_bp.set_attribute('fov','100')
        # camera_bp.set_attribute('slope','0.7')
        # camera position related to the vehicle
        camera_init_trans = carla.Transform(carla.Location(x=1.5, z=1.5))
        camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
        # output_path = os.path.join("../project/image", '%06d.png')
        
        if not os.path.exists(output_path): 
            os.makedirs(output_path)
        # camera.listen(lambda image: sensor_callback(image, sensor_queue, "camera"))
        cam_queue = Queue()
        camera.listen(cam_queue.put)
        # Create a queue to store and retrieve the sensor data
        sensor_list.append(camera)
        q_list.append(cam_queue)
        cam_idx = idx
        idx = idx+1
        print('RGB camera ready')

        # Spawn depth camera
        depth_bp = world.get_blueprint_library().find('sensor.camera.depth')
        depth_bp.set_attribute('fov','100')
        # depth_bp.set_attribute('slope','0.7')
        depth_init_trans = carla.Transform(carla.Location(x=1.5, z=1.5))
        # depth_bp.set_attribute('sensor_tick', str(tick_sensor))
        depth = world.spawn_actor(depth_bp, depth_init_trans, attach_to=vehicle)
        # cc_depth_log = carla.ColorConverter.LogarithmicDepth
        # nonvehicles_list.append(depth)
        depth_queue = Queue()
        depth_color_converter = carla.ColorConverter.LogarithmicDepth
        depth.listen(depth_queue.put)
        sensor_list.append(depth)
        q_list.append(depth_queue)
        depth_idx = idx
        idx = idx+1
        print('Depth camera ready')


        # Get the world to camera matrix
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
        # Get the attributes from the camera
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()
        # Calculate the camera projection matrix to project from 3D -> 2D
        K = build_projection_matrix(image_w, image_h, fov)

        # Retrieve all bounding boxes for traffic lights within the level
        # bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
        # Filter the list to extract bounding boxes within a 50m radius
        # nearby_bboxes = []
        # for bbox in bounding_box_set:
        #     if bbox.location.distance(camera.get_transform().location) < 50:
        #         nearby_bboxes
        
        # edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

        image_count = 0

        motorcycle_list = ["vehicle.harley-davidson.low_rider", "vehicle.kawasaki.ninja", "vehicle.vespa.zx125", "vehicle.yamaha.yzf"]
        bike_list = ["vehicle.bh.crossbike", "vehicle.diamondback.century", "vehicle.gazelle.omafiets"]
        emergency_list = ["vehicle.dodge.charger_police", "vehicle.dodge.charger_police_2020", "vehicle.carlamotors.firetruck", "vehicle.ford.ambulance"]

        while True:
            # Retrieve the image
            nowFrame = world.tick()

            data = [cva.retrieve_data(q,nowFrame) for q in q_list]
            assert all(x.frame == nowFrame for x in data if x is not None)

            # Skip if any sensor data is not available
            if None in data:
                continue
                
            vehicles_raw = world.get_actors().filter('vehicle.*')
            snap = data[tick_idx]
            image = data[cam_idx]
            depth_image = data[depth_idx]

            # image = cam_queue.get(block = True)
            # depth_image = depth_queue.get(block = True)

            img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

            # Get the camera matrix 
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

            # Save image
            if image_count % 10 == 0:
                image.save_to_disk(os.path.join(output_path, '%06d.png' % image.frame))
                # depth_image.save_to_disk(os.path.join(output_path, '%06d_d.png' % image.frame), depth_color_converter)
                open(os.path.join(output_path, f"{image.frame}.txt"), "a")
            # (PASCAL VOC format) Initialize the exporter
            # writer = Writer(output_path + '.png', image_w, image_h)


            boundingbox_path = os.path.join(output_path, "boundingbox")
            if not os.path.exists(boundingbox_path): 
                os.makedirs(boundingbox_path)

            for npc in world.get_actors().filter('*vehicle*'):
                annotation_str = ""
                if npc.id != vehicle.id:
                    bb = npc.bounding_box
                    dist = npc.get_transform().location.distance(vehicle.get_transform().location)
                    if dist < 75:
                        forward_vec = vehicle.get_transform().get_forward_vector()
                        ray = npc.get_transform().location - vehicle.get_transform().location
                        if forward_vec.dot(ray) > 1:
                            verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                            x_max = -10000
                            x_min = 10000
                            y_max = -10000
                            y_min = 10000
                            for vert in verts:
                                p = get_image_point(vert, K, world_2_camera)
                                if p[0] > x_max:
                                    x_max = p[0]
                                if p[0] < x_min:
                                    x_min = p[0]
                                if p[1] > y_max:
                                    y_max = p[1]
                                if p[1] < y_min:
                                    y_min = p[1]

                            # filter_occlusion_bbox(dist, verts, )
                            dist_margin = 10000
                            for vert in verts:
                                dist_vert = vehicle.get_transform().location.distance(vert)
                                dist_margin = min(dist_margin, dist_vert)

                            xc = int((x_max+x_min)/2)
                            yc = int((y_max+y_min)/2)
                            wp = int((x_max-x_min) * 0.8/2)
                            hp = int((y_max-y_min) * 0.8/2)
                            u1 = xc-wp
                            u2 = xc+wp
                            v1 = yc-hp
                            v2 = yc+hp

                            draw_boundingbox(img, u1, v1, u2, v2, color=(255,0,0,255))

                            depth_meter = cva.extract_depth(depth_image)
                            depth_bb = np.array(depth_meter[v1:v2+1,u1:u2+1])
                            
                            dist_delta_new = np.full(depth_bb.shape, dist - 8)
                            # print("depth_bb:", depth_bb)
                            # print("dist_delta_new:", dist_margin)
                            s_patch = np.array(depth_bb > dist_delta_new)
                            s = np.sum(s_patch) > s_patch.shape[0]*0.5

                            # (PASCAL VOC format) Add the object to the frame (ensure it is inside the image)
                            # if x_min > 0 and x_max < image_w and y_min > 0 and y_max < image_h: 
                            #     writer.addObject('vehicle', x_min, y_min, x_max, y_max)
                            if s == True:
                                draw_boundingbox(img, x_min, y_min, x_max, y_max)

                                # Add the object to the frame (ensure it is inside the image)
                                if x_min > 0 and x_max < image_w and y_min > 0 and y_max < image_h: 
                                    if npc.type_id in bike_list:
                                        class_id = 0
                                    elif npc.type_id in motorcycle_list:
                                        class_id = 1
                                    elif npc.type_id in emergency_list:
                                        class_id = 2
                                    else:
                                        class_id = 3
                                    x_center = ((x_min + x_max) / 2) / image_w
                                    y_center = ((y_min + y_max) / 2) / image_h
                                    width = (x_max - x_min) / image_w
                                    height = (y_max - y_min) / image_h
                                    annotation_str += f"{class_id} {x_center} {y_center} {width} {height}\n"
                                    
                                    if image_count % 10 == 0:
                                        with open(os.path.join(output_path, f"{image.frame}.txt"), "a") as f:
                                            f.write(annotation_str)

            
            bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)

            print("---------------------------------------------------")
            for bb in bounding_box_set:
                annotation_str = ""
                # Filter for distance from ego vehicle
                dist = bb.location.distance(vehicle.get_transform().location)
                if  dist < 75:

                    # Calculate the dot product between the forward vector
                    # of the vehicle and the vector between the vehicle
                    # and the bounding box. We threshold this dot product
                    # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
                    forward_vec = vehicle.get_transform().get_forward_vector()
                    ray = bb.location - vehicle.get_transform().location

                    if forward_vec.dot(ray) > 1:
                        # Cycle through the vertices
                        verts = [v for v in bb.get_world_vertices(carla.Transform())]
                        x_max = -10000
                        x_min = 10000
                        y_max = -10000
                        y_min = 10000                        
                        for vert in verts:
                            # Join the vertices into edges
                            p = get_image_point(vert, K, world_2_camera)
                            if p[0] > x_max:
                                x_max = p[0]
                            if p[0] < x_min:
                                x_min = p[0]
                            if p[1] > y_max:
                                y_max = p[1]
                            if p[1] < y_min:
                                y_min = p[1]

                        dist_margin = 10000
                        for vert in verts:
                            dist_vert = vehicle.get_transform().location.distance(vert)
                            dist_margin = min(dist_margin, dist_vert)

                        xc = int((x_max+x_min)/2)
                        yc = int((y_max+y_min)/2)
                        wp = int((x_max-x_min) * 0.8/2)
                        hp = int((y_max-y_min) * 0.8/2)
                        u1 = xc-wp
                        u2 = xc+wp
                        v1 = yc-hp
                        v2 = yc+hp

                        draw_boundingbox(img, u1, v1, u2, v2, color=(255,0,0,255))

                        depth_meter = cva.extract_depth(depth_image)
                        depth_bb = np.array(depth_meter[v1:v2+1,u1:u2+1])                            
                            
                        dist_delta_new = np.full(depth_bb.shape, dist - 10)
                        s_patch = np.array(depth_bb > dist_delta_new)
                        print("dist", dist - 10)
                        print("depth_bb: ")
                        print(depth_bb)
                        print("u1: ", u1, ", u2: ", u2, ", v1: ", v1, ", v2: ", v2)
                        s = np.sum(s_patch) > s_patch.shape[0]*0.1

                        if s:
                            # Draw the edges into the camera output
                            draw_boundingbox(img, x_min, y_min, x_max, y_max)

                            if x_min > 0 and x_max < image_w and y_min > 0 and y_max < image_h: 
                                class_id = 4
                                x_center = ((x_min + x_max) / 2) / image_w
                                y_center = ((y_min + y_max) / 2) / image_h
                                width = (x_max - x_min) / image_w
                                height = (y_max - y_min) / image_h
                                annotation_str += f"{class_id} {x_center} {y_center} {width} {height}\n"
                                    
                            if image_count % 10 == 0:
                                with open(os.path.join(output_path, f"{image.frame}.txt"), "a") as f:
                                    f.write(annotation_str)


            # Show image with bounding box
            cv2.imshow('ImageWindowName',img)
            # Save image with bounding box
            if image_count % 10 == 0:
                output_file_path = os.path.join(boundingbox_path, f"{image.frame}_b.png")
                cv2.imwrite(output_file_path, img)
            image_count += 1
            if cv2.waitKey(1) == ord('q'):
                break

            # (PASCAL VOC format) Save the bounding boxes in the scene
            # writer.save(os.path.join(output_path, '%06d.xml' % image.frame))

        cv2.destroyAllWindows()

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