import os, shutil,sys
import pandas as pd
import numpy as np 
from skimage.draw import polygon
import json
from skimage import transform
from sklearn.metrics import jaccard_score

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

def getLabels(ids, y_directory,keepLabels=keeplabels, road=False):
    labels = []
    if road:
        keeplabels['road']=7
    for id in ids:
        lb_path = os.path.join(y_directory, id+'.json')
        labels.append(getInstances(lb_path,keepLabels))
    return labels

def generateCSV(filepath:str, ids, labels,obstruction_label=None):
    if obstruction_label:
        row = zip(ids,labels,obstruction_label)
        cols = ['ID','LABEL','OBSTRUCTION']
    row = zip(ids,labels)
    cols = ['ID','LABEL']
    df = pd.DataFrame(data=list(row),columns=cols)
    df.to_csv(filepath, index=False)

def retrieve_camera(filename):
    '''returns baseline and focal length of camera calibration files'''
    with open(filename) as f:
        data = json.load(f)
        return data['extrinsic']['baseline'], data['intrinsic']['fx']
    
def disparity_value(d):
    d[d > 0] = (d[d > 0] - 1) / 256
    return d

def calculate_depth(disparity, baseline, focal_length):
    disparity = disparity_value(disparity)
    mask = (disparity!=0)
    depth = np.empty_like(disparity)
    depth[mask] = (baseline*focal_length)/disparity[mask]
    depth[~mask] = 0
    return depth

def parse_json(filename: str, keep=keeplabels, return_labels = False, int_labels=False, resize:tuple =None, road=False):
    polygons = []
    if road:
        keeplabels['road']=7
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
def create_mask(path_to_mask, labels,resize:tuple=None):
    
    im_dim,polygons = parse_json(path_to_mask, 
                                 return_labels=True, 
                                 int_labels=True, 
                                 resize=resize)
    key_count = {}
    masks = []
    for label in labels:
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

def stack_mask(id, data,resize:tuple=None,dir=LBL_DIR, road=False):
    labels = data[id]
    im_dim,polygons = parse_json(dir+f'/{id}.json', return_labels=True, int_labels=True, resize=resize, road=road)
    key_count = {}
    masks = []
    for label in labels:
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

def generate_ious(labels, masks,sidewalk_lbl=8, IOU_THRESHOLD=0.001, road_lbl=None):
    '''
    Generates the ious for all the segmentations against all instances 
    of sidewalk masks.

    Returns dictionary of format {label_index: (label_index, ious_score)}
    '''
    sidewalk_idx = np.where(labels==sidewalk_lbl)[0]
    if road_lbl:
        road_idx = np.where(labels==road_lbl)[0]

    all_labels = np.unique(labels)
    ious = {}

    for label in all_labels:
        # ignore calculating sidewalks and raods
        if label == sidewalk_lbl: continue
        if road_lbl and label==road_lbl: continue

        for sidewalk in sidewalk_idx:
            sidewalk_mask = masks[:,:,sidewalk]
            non_sidewalk = np.where(labels==label)[0]
            for i in non_sidewalk:
                iou = jaccard_score(sidewalk_mask, masks[:,:,i],average='micro',zero_division={0.0, 1.0})
                
                # if IOU(car, road)>0: ignore this car
                if road_lbl and labels[i]==26:
                    for road in road_idx:
                        road_mask = masks[:,:,road]
                        iou = jaccard_score(road_mask, masks[:,:,i],average='micro',zero_division={0.0, 1.0})
                        # print(i,iou)
                        if iou>0:
                            iou =0

                if iou >= IOU_THRESHOLD:
                    if i not in ious:
                        ious[i] = (sidewalk, iou)
                    else:
                        if ious[i][-1] < iou:
                            ious[i] = (sidewalk,iou)
    return ious

def generate_captions_obstructed(labels, ious, obs_labels=None, labels_dict=inv_keeplabels, IOU_THRESHOLD = 0.002):
    '''
    Caption generation for easy viewing of obstruction
    '''
    captions = []
    for i in range(len(labels)):
        tmp = ''
        if i in ious: 
            tmp += f'{i} {labels_dict[labels[i]]}'
            sidewalk, iou = ious[i]
            if iou>=IOU_THRESHOLD:
                tmp += f'({sidewalk}) {round(iou,3)}'
        captions.append(tmp)
    return captions
        

def generate_obstruction_from_IOUS():
    pass