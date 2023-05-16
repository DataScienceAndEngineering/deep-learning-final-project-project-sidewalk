from collections import namedtuple
import os, shutil,sys
import pandas as pd
import numpy as np 
from skimage.draw import polygon
import json
from skimage import transform
from sklearn.metrics import jaccard_score
import random

ROOT_DIR = os.path.abspath("../../")
PROCESSED_DATA = os.path.join(ROOT_DIR, 'data/processed')
sys.path.append(ROOT_DIR)  # To find local version of the library

IMG_DIR = os.path.join(PROCESSED_DATA, 'images')
LBL_DIR = os.path.join(PROCESSED_DATA, 'segmentations')

keeplabels ={'sidewalk':8,
            'rail track':10,
            'wall':12,
            'fence':13,
            'guard rail':14,
            'pole':17,
            'polegroup':18,
            'vegetation': 21,
            'car':26,
            'bicycle':33}
            
inv_keeplabels =  {
    8: 'sidewalk',
    10: 'rail track',
    12: 'wall',
    13: 'fence',
    14: 'guard rail',
    17: 'pole',
    18:  'polegroup',
    21: 'vegetation',
    26: 'car',
    33: 'bicycle'}


# helper function to help preprocess the data 
# and parse through the dataset
def moveUp(old_directory, new_directory=None, to_keep = None, mask=False):
    '''
    part of preprocessing - moving
    '''
    for cities in os.listdir(old_directory):
        subdir = os.path.join(old_directory,cities)
        if cities == '.DS_Store':
            continue
        for img in os.listdir(subdir):
            _id = img.split('.')
            if mask and _id[1]!='json':
                continue
            id = "_".join(_id[0].split("_")[:3])
            if id in to_keep:
                img_path = os.path.join(subdir,img)
                if mask:
                    new_path = os.path.join(new_directory,'segmentations',f'{id}.json')
                else:
                    new_path = os.path.join(new_directory,'images',f'{id}.json')
                
                shutil.move(img_path, new_path)


def getInstances(filename:str, labelsToKeep:dict):
    labels = ''
    # labels = []
    keys = labelsToKeep.keys()
    data = json.load(open(filename))
    data = data["objects"]
    for d in data:
        l = d['label']
        if l in keys:
            # labels.append(labelsToKeep[l])
            labels += str(labelsToKeep[l])+' '
    return labels

def getLabels(ids, y_directory,keepLabels=keeplabels):
    labels = []
    for id in ids:
        lb_path = os.path.join(y_directory, id+'.json')
        labels.append(getInstances(lb_path,keepLabels))
    return labels

def generateCSV(filepath:str, ids, labels):
    df = pd.DataFrame(data=list(zip(ids, labels)),columns=['ID','LABEL'])
    df.to_csv(filepath, index=False)


def parse_json(filename: str, keep=keeplabels, return_labels = False, int_labels=False, resize:tuple =None):
    polygons = []
    if return_labels:
        label_n_polygons = {}
    with open(filename) as f:
        data = json.load(f)
        im_h, im_w = data['imgHeight'], data['imgWidth']
        dim = (im_h,im_w)
        if resize:
            dim = resize
        objects = data["objects"]
        for i in objects:
            # print(i['label'])
            lab = i['label']
            if lab in keep:

                poly_coords = i['polygon']
                tmp = list(zip(*poly_coords))
                y,x = np.array(tmp[0]), np.array(tmp[1])
                # tmp[0],tmp[1] =  np.array(tmp[0]), np.array(tmp[1])
                if resize:
                    r_h,r_w = resize[0]/im_h ,resize[1]/im_w
                    y,x = y*r_h, x*r_w
                
                    
                if return_labels:
                    if int_labels:
                        lab = keep[lab]
                    if lab not in label_n_polygons:
                        label_n_polygons[lab] = []
                    label_n_polygons[lab].append((y,x))
                else: 
                    polygons.append((y, x))
    # return (im_h, im_w), labels, polygons

    if return_labels:
        return dim, label_n_polygons
    return dim, polygons

# https://stackoverflow.com/questions/72168663/setting-a-list-of-x-y-cooordinates-into-an-array-so-polygons-are-drawn
def draw_instance_Mask(img,poly):
    
    r = poly[0]
    c = poly[1]
    r_index, c_index = polygon(r, c)
    r_index, c_index = np.clip(r_index,0,img.shape[1]-1),np.clip(c_index,0,img.shape[0]-1)
    # print(type(r_index),max(r_index),max(c_index))

    # r_index = [i if i <imgsize[0] else imgsize[0]-1 for i in r_index]
    # c_index = [i if i <imgsize[1] else imgsize[1]-1 for i in c_index]
    # new_index = max(0, min(new_index, len(mylist)-1))

    img[c_index,r_index] = 1
    # return img


def draw_flat_mask(img, polygons):
    for poly in polygons:
        if isinstance(polygons, dict):
            ps = polygons[poly]
            for p in ps:
                draw_instance_Mask(img, p)
        else:
            draw_instance_Mask(img, poly)

def create_mask(path_to_mask, labels, resize:tuple=None, tmp_mapping=None):
    im_dim,polygons = parse_json(path_to_mask, 
                                 return_labels=True, 
                                 int_labels=True, 
                                 resize=resize)
    key_count = {}
    masks = []
    for label in labels:
        if tmp_mapping: 
            label = tmp_mapping[label]
        if label not in key_count:
            key_count[label]=0
        count = key_count[label]
        mask = polygons[label][count]
        img = np.zeros(im_dim, dtype=np.uint8)
        draw_instance_Mask(img,mask)
        masks.append(img)
        key_count[label]+=1
    
    masks = np.stack(masks, axis=-1)
    return masks

def generate_ious(labels, masks,sidewalk_lbl=8, IOU_THRESHOLD=0):
    '''
    Generates the ious for all the segmentations against all instances 
    of sidewalk masks.

    Returns dictionary of format {label_index: (label_index, ious_score)}
    '''
    sidewalk_idx = np.where(labels==sidewalk_lbl)[0]
    # sidewalk_masks = masks[:,:, sidewalk_idx]

    all_labels = np.unique(labels)
    ious = {}

    for label in all_labels:
        if label == sidewalk_lbl: continue
        for sidewalk in sidewalk_idx:
            sidewalk_mask = masks[:,:,sidewalk]
            non_sidewalk = np.where(labels==label)[0]
            for i in non_sidewalk:
                iou = jaccard_score(sidewalk_mask, masks[:,:,i],average='micro',zero_division={0.0, 1.0})
                if iou > IOU_THRESHOLD:
                    if i not in ious:
                        ious[i] = (sidewalk, iou)
                    else:
                        if ious[i][-1] < iou:
                            ious[i] = (sidewalk,iou)
    return ious

def generate_captions_obstructed(labels, ious, labels_dict=inv_keeplabels):
    '''
    Caption generation for easy viewing of obstruction
    '''
    captions = []
    for i in range(len(labels)):
        tmp = ''
        if i in ious: 
            sidewalk, iou = ious[i]

            if round(iou,3)>0:
                tmp += f'{i} {labels_dict[labels[i]]}'
                tmp += f'({sidewalk}) {round(iou,3)}'
        captions.append(tmp)
    return captions
        




# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
# Label = namedtuple( 'Label' , [

#     'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
#                     # We use them to uniquely name a class

#     'id'          , # An integer ID that is associated with this label.
#                     # The IDs are used to represent the label in ground truth images
#                     # An ID of -1 means that this label does not have an ID and thus
#                     # is ignored when creating ground truth images (e.g. license plate).
#                     # Do not modify these IDs, since exactly these IDs are expected by the
#                     # evaluation server.

#     'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
#                     # ground truth images with train IDs, using the tools provided in the
#                     # 'preparation' folder. However, make sure to validate or submit results
#                     # to our evaluation server using the regular IDs above!
#                     # For trainIds, multiple labels might have the same ID. Then, these labels
#                     # are mapped to the same class in the ground truth images. For the inverse
#                     # mapping, we use the label that is defined first in the list below.
#                     # For example, mapping all void-type classes to the same ID in training,
#                     # might make sense for some approaches.
#                     # Max value is 255!

#     'category'    , # The name of the category that this label belongs to

#     'categoryId'  , # The ID of this category. Used to create ground truth images
#                     # on category level.

#     'hasInstances', # Whether this label distinguishes between single instances or not

#     'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
#                     # during evaluations or not

#     'color'       , # The color of this label
#     ] )

# labels = [
#     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#     Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
#     Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
#     Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
#     Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
#     Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
#     Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
#     Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
#     Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
#     Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
#     Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
#     Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
#     Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
#     Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
#     Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
#     Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
#     Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
#     Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
#     Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
#     Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
#     Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
#     Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
#     Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
#     Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
#     Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
#     Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
#     Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
#     Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
#     Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
#     Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
#     Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
# ]