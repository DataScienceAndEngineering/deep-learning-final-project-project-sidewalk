#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 13:05:25 2023

@author: nicholas
"""
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

json_path = '/home/nicholas/Documents/Project_Sidewalk/Project_Sidewalk/gtFine/train/aachen/aachen_000170_000019_gtFine_polygons.json'
img_path = '/home/nicholas/Documents/Project_Sidewalk/Project_Sidewalk/leftImg8bit/train/aachen/aachen_000170_000019_leftImg8bit.png'
image = Image.open(img_path)
plt.imshow(image)
instance_data = json.load(open(json_path))
label_keep = {
    'sidewalk': 8,
    'pole': 17,
    'car': 26,
    'bicycle': 33
    }
img = Image.new('L', (2048, 1024), 0)
j=1
img_map = {}
for i in instance_data['objects']:
    if i['label'] in label_keep.keys():
        polygon = i['polygon']
        xy = [(p[0], p[1]) for p in polygon]
        ImageDraw.Draw(img).polygon(xy,fill=j)
        img_map[j] = i['label']
        j+=1
k =0
extracted_car = np.empty((1024, 2048, 3))
for i in img_map.values():
    if i == 'car':
        idx = (np.array(img)==list(img_map.keys())[k])
        if idx.sum() > 12000:
            extracted_car[idx] = np.array(image)[idx]
            break
    k+=1

        
nonzero_indices = np.nonzero(extracted_car)
min_indices = np.min(nonzero_indices, axis=1)
max_indices = np.max(nonzero_indices, axis=1)
# Crop the object array to remove padding zeros
cropped = extracted_car[min_indices[0]:max_indices[0]+1, min_indices[1]:max_indices[1]+1, ...]

cropped = np.clip(cropped, 0, 255).astype(np.uint8)
plt.imshow(cropped)

car_size = cropped.shape[:-1]
k=0
sidewalks=list()
idx_sums = list()
idx = list()
for i in img_map.values():
    if i == 'sidewalk':
        mask = np.array(img) == list(img_map.keys())[k]
        sidewalks.append(mask)
        idx_sums.append(np.sum(mask))
    k+=1
largest_walk = sidewalks[np.argwhere(idx_sums == np.max(idx_sums))[0][0]]

y, x = np.where(largest_walk)
median_location = np.median(np.column_stack((y, x)), axis=0)
median_0 = int(median_location[0])
median_1 = int(median_location[1])
print(car_size)
car_1 = car_size[1]
car_0 = car_size[0]
while car_1+median_1 > 2048:
    car_1 -= 1
while median_0-car_0 < 0:
    car_0 -= 1
np_image = np.array(image)
cropped = cropped[0:car_0, 0:car_1]
np_image[median_0-car_0:median_0, median_1:median_1+car_1] = cropped
for i in range(median_0-car_0,median_0):
    for j in range(median_1,median_1+car_1):
        if (np_image[i,j] == 0).all():
            np_image[i,j] = np.array(image)[i,j]
plt.imshow(cropped)
plt.imshow(np_image)