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
import os
import random

def build_masks(instance_data, label_keep):
    #converts json data into masks for all labels requested
    img = Image.new('L', (instance_data['imgWidth'], instance_data['imgHeight']), 0)
    img_map = {}
    j=1
    for i in instance_data['objects']:
        if i['label'] in label_keep.keys():
            polygon = i['polygon']
            xy = [(p[0], p[1]) for p in polygon]
            ImageDraw.Draw(img).polygon(xy,fill=j)
            img_map[j] = i['label']
            j+=1
    return img, img_map

def isolate_mask(img, img_map, label='pole', thresh=2000):
    #Isolates a mask of label object over a provided threshold size
    k = 0
    extracted_car = np.empty((np.array(img).shape[0], np.array(img).shape[1], 3))
    for i in img_map.values():
        if i == label:
            idx = (np.array(img)==list(img_map.keys())[k])
            if idx.sum() > thresh:
                extracted_car[idx] = np.array(image)[idx]
                return extracted_car
        k+=1
    raise Exception(f"No mask for {label} found over threshold of {thresh}")

def crop(image):
    #Removes all padding zeros, cropping to smallest size
    nonzero_map = np.nonzero(image)
    min_indices = np.min(nonzero_map, axis=1)
    max_indices = np.max(nonzero_map, axis=1)
    cropped = image[min_indices[0]:max_indices[0]+1, min_indices[1]:max_indices[1]+1, :]
    return np.clip(cropped, 0, 255).astype(np.uint8)

def locate_dest(img, img_map):
    sidewalks = list()
    idx_sums = list()
    k=0
    for i in img_map.values():
        if i == 'sidewalk':
            sidewalks.append(np.array(img) == list(img_map.keys())[k])
            idx_sums.append(np.sum(sidewalks[-1])) 
        k+=1
    largest = sidewalks[np.argwhere(idx_sums == np.max(idx_sums))[0][0]]
    x, y = np.where(largest)
    #median_location = np.median(np.column_stack((y,x)),axis=0)
    median_location = [np.median(np.unique(y)), np.median(np.unique(x))]
    return (median_location[1],median_location[0])

def paste(base, img, destination):
    # Crop the object array to remove padding zeros
    [size_0, size_1] = img.shape[:-1]
    median_0 = int(destination[0])
    median_1 = int(destination[1])
    while size_1+median_1 > 2048:
        size_1 -= 1
    while median_0-size_0 < 0:
        size_0 -= 1
    base = np.array(base)
    orig = np.array(base)
    cropped = img[0:size_0, 0:size_1]
    base[median_0-size_0:median_0, median_1:median_1+size_1,:] = cropped
    for i in range(median_0-size_0,median_0):
        for j in range(median_1,median_1+size_1):
            if (np.sum(base[i,j,:]) == 0):
                base[i,j,:] = orig[i,j,:]
    return base

def json2img(file):
    return file.replace('_gtFine_polygons.json', '_leftImg8bit.png')

json_path = '/home/nicholas/Documents/Project_Sidewalk/Project_Sidewalk/json/train/'
img_path = '/home/nicholas/Documents/Project_Sidewalk/Project_Sidewalk/images_RAW/train/'
jsons = os.listdir(json_path)
imgs = os.listdir(img_path)

json_file = random.choice(jsons)
img_file = json2img(json_file)
json_fpath = json_path+json_file
img_fpath = img_path+img_file
image = Image.open(img_fpath)
instance_data = json.load(open(json_fpath))
label_keep = {
    'sidewalk': 8,
    'pole': 17,
    'car': 26,
    'bicycle': 33
    }
       
img, img_map = build_masks(instance_data, label_keep)
extracted_car = isolate_mask(img, img_map, label='car', thresh=12000)
car = crop(extracted_car)
destination = locate_dest(img, img_map)     
output = paste(image, car, destination)

plt.imshow(car)
plt.imshow(output)