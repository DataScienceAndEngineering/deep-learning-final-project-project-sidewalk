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
import string

class augmenter:
    
    def json2img(self,file):
        return file.replace('_gtFine_polygons.json', '_leftImg8bit.png')
             
    def __init__(self, image_path, json_path, out_path):
        self.image_path = image_path
        self.json_path = json_path
        self.out_path = out_path
        
        if not os.path.isdir(self.out_path):
            os.mkdir(self.out_path)
        if not os.path.isdir(self.out_path+'images/'):
            os.mkdir(self.out_path+'images/')
        if not os.path.isdir(self.out_path+'json/'):
            os.mkdir(self.out_path+'json/')


        self.label_keep = {
            'sidewalk': 8,
            'pole': 17,
            'car': 26,
            'bicycle': 33
            }
        self.image_files = os.listdir(self.image_path)
        self.json_files = os.listdir(self.json_path)
        self.label = 'car'
        self.thresh = 12000
        self.new()
        
    def build_masks(self):
        #converts json data into masks for all labels requested
        self.masks = Image.new('L', (self.instance_data['imgWidth'], self.instance_data['imgHeight']), 0)
        self.map = {}
        j=1
        self.reduced_instance = list()
        for i in self.instance_data['objects']:
            if i['label'] in self.label_keep.keys():
                self.reduced_instance.append(i)
                polygon = i['polygon']
                xy = [(p[0], p[1]) for p in polygon]
                ImageDraw.Draw(self.masks).polygon(xy,fill=j)
                self.map[j] = i['label']
                j+=1
                
    def update_label(self, label):
        self.label = label
        self.isolate_mask()
        self.crop()
        self.paste()
        self.preview()
        
    def update_thresh(self, thresh):
        self.thresh = thresh
        self.isolate_mask()
        self.crop()
        self.paste()
        self.preview()
        
    def isolate_mask(self):
        #Isolates a mask of label object over a provided threshold size
        k = 0
        extracted = np.empty((np.array(self.masks).shape[0], np.array(self.masks).shape[1], 3))
        self.all_extracted = list()
        self.instance = list()
        np_image = np.array(self.image).astype(int)
        for i in self.map.values():
            if i == self.label:
                idx = (np.array(self.masks)==list(self.map.keys())[k])
                if idx.sum() > self.thresh:
                    self.instance.append(self.reduced_instance[k])
                    extracted = np.empty((np.array(self.masks).shape[0], np.array(self.masks).shape[1], 3)).astype(int)
                    extracted[idx] = np_image[idx]
                    self.all_extracted.append(extracted.astype(int))
            k+=1
        if len(self.all_extracted) == 0:
            raise Exception(f"No mask for {self.label} found over threshold of {self.thresh}")
        self.active_extraction = self.all_extracted[self.index]

            
    def forward(self):
        #Next extracted object
        try:
            self.active_extraction = self.all_extracted[self.index+1]
            self.crop()
            self.paste()
            self.preview()
            self.index += 1
        except: raise Exception('Last mask shown')
        
    def previous(self):
        #Previous extracted object
        try: 
            self.active_extraction = self.all_extracted[self.index-1]
            self.crop()
            self.paste()
            self.preview()
            self.index -= 1
        except: raise Exception('First mask shown')
        
    def new(self):
        #New random image sample from path
        selection = random.choice(self.json_files)
        self.instance_data = json.load(open(self.json_path + selection))
        self.image = Image.open(self.image_path + self.json2img(selection))
        self.index = 0 
        self.build_masks()
        self.isolate_mask()
        self.crop()
        self.locate_dest()
        self.paste()
        self.preview()
        
    def crop(self):
        #Removes all padding zeros, cropping to smallest size
        nonzero_map = np.nonzero(self.active_extraction)
        min_indices = np.min(nonzero_map, axis=1)
        max_indices = np.max(nonzero_map, axis=1)
        self.cropped = self.active_extraction[min_indices[0]:max_indices[0]+1, min_indices[1]:max_indices[1]+1, :]
        #return np.clip(cropped, 0, 255).astype(np.uint8)

    def locate_dest(self):
        sidewalks = list()
        idx_sums = list()
        k=0
        for i in self.map.values():
            if i == 'sidewalk':
                sidewalks.append(np.array(self.masks) == list(self.map.keys())[k])
                idx_sums.append(np.sum(sidewalks[-1])) 
            k+=1
        largest = sidewalks[np.argwhere(idx_sums == np.max(idx_sums))[0][0]]
        x, y = np.where(largest)
        #median_location = np.median(np.column_stack((y,x)),axis=0)
        self.paste_loc = [np.median(np.unique(x)), np.median(np.unique(y))]

    def paste(self):
        # Crop the object array to remove padding zeros
        self.output = np.array(self.image)
        [size_0, size_1] = self.cropped.shape[:-1]
        median_0 = int(self.paste_loc[0])
        median_1 = int(self.paste_loc[1])
        while size_1+median_1 > 2048:
            size_1 -= 1
        while median_0-size_0 < 0:
            size_0 -= 1
        self.new_mask = np.zeros(shape=self.output.shape)
        orig = np.array(self.output)
        self.cropped_tight = self.cropped[0:size_0, 0:size_1]
        self.output[median_0-size_0:median_0, median_1:median_1+size_1,:] = self.cropped_tight
        self.new_mask[median_0-size_0:median_0, median_1:median_1+size_1,:] = 1
        for i in range(median_0-size_0,median_0):
            for j in range(median_1,median_1+size_1):
                if (np.sum(self.output[i,j,:]) == 0):
                    self.output[i,j,:] = orig[i,j,:]
                    self.new_mask[i,j,:] = 0

    def preview(self):
        #Displays original, extracted object, and augmented samples
        plt.figure(figsize=(30,10))
        plt.subplot(3,1,1)
        plt.imshow(self.image)
        plt.axis('off')
        plt.subplot(3,1,2)
        plt.imshow(self.cropped_tight)
        plt.axis('off')
        plt.subplot(3,1,3)
        plt.imshow(self.output)
        plt.axis('off')
        
    def append_json(self):
        orig_def = self.instance[self.index]
        distance = (np.median(np.argwhere(self.new_mask !=0), axis=0) -np.median(np.argwhere(self.active_extraction != 0), axis=0))[:-1]
        distance = [int(distance[1]), int(distance[0])]
        print(distance)
        size = [2048, 1024]
        self.new_def = {
            'label': self.label,
            'polygon': list() 
            }
        for i in orig_def['polygon']:
            for j in [0, 1]:
                if i[j] + distance[j] < 0:
                    i[j] = 0
                elif i[j] + distance[j] > size[j]:
                    i[j] = int(size[j])
                else:
                   i[j] += int(distance[j])
            self.new_def['polygon'].append(i)
        self.instance_data['objects'].append(self.new_def)
        
    def test_newjson(self):
        polygon = self.new_def['polygon']
        xy = [(p[0], p[1]) for p in polygon]
        new = Image.new('L', (tool.instance_data['imgWidth'], tool.instance_data['imgHeight']), 0)
        ImageDraw.Draw(new).polygon(xy,fill=1)
        plt.imshow(new)
            
    def save(self):
        output_image = Image.fromarray(self.output)
        d = ''
        for i in range(0,17):
            d += str(random.randint(0,9))
        name ='aug_' + d + '_leftImg8bit.png'
        output_image.save(self.out_path+'images/'+name)
        name = 'aug_' + d + '_gtFine_polygons.json'
        self.append_json()
        with open(self.out_path+'json/'+name, "w") as outfile:
            json.dump(self.instance_data, outfile)
        self.new()
        

# json_path = '/home/nicholas/Documents/Project_Sidewalk/Project_Sidewalk/json/train/'
# img_path = '/home/nicholas/Documents/Project_Sidewalk/Project_Sidewalk/images_RAW/train/'
# out_path = '/home/nicholas/Documents/Project_Sidewalk/Project_Sidewalk/augmented/'
# tool = augmenter(img_path, json_path, out_path)
