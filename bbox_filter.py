import glob
import os
import sys
import carla
import math
import random
from queue import Queue
from queue import Empty
import numpy as np

def extract_depth(depth_img):
    depth_img.convert(carla.ColorConverter.Depth)
    depth_meter = np.array(depth_img.raw_data).reshape((depth_img.height,depth_img.width,4))[:,:,0] * 1000 / 255
    return depth_meter

def retrieve_data(sensor_queue, frame, timeout=5):
    while True:
        try:
            data = sensor_queue.get(True,timeout)
        except Empty:
            return None
        if data.frame == frame:
            return data

# def auto_annotate(vehicles, camera, depth_img, max_dist=100, depth_margin=-1, patch_ratio=0.5, resize_ratio=0.5, json_path=None):
#     depth_show = False
#     vehicles = filter_angle_distance(vehicles, camera, max_dist)
#     bounding_boxes_2d = [get_2d_bb(vehicle, camera) for vehicle in vehicles]
#     if json_path is not None:
#         vehicle_class = get_vehicle_class(vehicles, json_path)
#     else:
#         vehicle_class = []
#     filtered_out, removed_out, _, _ = filter_occlusion_bbox(bounding_boxes_2d, vehicles, camera, depth_img, vehicle_class, depth_show, depth_margin, patch_ratio, resize_ratio)
#     return filtered_out, removed_out 

# def filter_occlusion_bbox(bounding_boxes, vehicles, sensor, depth_img, v_class=None, depth_capture=False, depth_margin=-1, patch_ratio=0.5, resize_ratio=0.5):
#     filtered_bboxes = []
#     filtered_vehicles = []
#     filtered_v_class = []
#     filtered_out = {}
#     removed_bboxes = []
#     removed_vehicles = []
#     removed_v_class = []
#     removed_out = {}
#     selector = []
#     patches = []
#     patch_delta = []
#     _, v_transform_s = get_list_transform(vehicles, sensor)
    
#     for v, vs, bbox in zip(vehicles,v_transform_s,bounding_boxes):
#         dist = vs[:,0]

#         if depth_margin < 0:
#             depth_margin = (v.bounding_box.extent.x**2+v.bounding_box.extent.y**2)**0.5 + 0.25
        
#         uc = int((bbox[0,0]+bbox[1,0])/2)
#         vc = int((bbox[0,1]+bbox[1,1])/2)
#         wp = int((bbox[1,0]-bbox[0,0])*resize_ratio/2)
#         hp = int((bbox[1,1]-bbox[0,1])*resize_ratio/2)
#         u1 = uc-wp
#         u2 = uc+wp
#         v1 = vc-hp
#         v2 = vc+hp
#         depth_patch = np.array(depth_img[v1:v2,u1:u2])

#         dist_delta = dist-depth_patch
#         s_patch = np.array(dist_delta < depth_margin)
#         s = np.sum(s_patch) > s_patch.shape[0]*patch_ratio
#         selector.append(s)
#         patches.append(np.array([[u1,v1],[u2,v2]]))
#         patch_delta.append(dist_delta)
    
#     for bbox,v,s in zip(bounding_boxes,vehicles,selector):
#         if s:
#             filtered_bboxes.append(bbox)
#             filtered_vehicles.append(v)
#         else:
#             removed_bboxes.append(bbox)
#             removed_vehicles.append(v)

#     filtered_out['bbox']=filtered_bboxes
#     filtered_out['vehicles']=filtered_vehicles
#     removed_out['bbox']=removed_bboxes
#     removed_out['vehicles']=removed_vehicles
        
#     if v_class is not None:
#         for cls,s in zip(v_class,selector):
#             if s:
#                 filtered_v_class.append(cls)
#             else:
#                 removed_v_class.append(cls)
#         filtered_out['class']=filtered_v_class
#         removed_out['class']=removed_v_class
    
#     if depth_capture:
#         depth_debug(patches, depth_img, sensor)
#         for i,matrix in enumerate(patch_delta):
#             print("\nvehicle "+ str(i) +": \n" + str(matrix))
#         depth_capture = False
        
#     return filtered_out, removed_out, patches, depth_capture