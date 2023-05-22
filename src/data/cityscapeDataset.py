import os,sys

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library


from models.maskrcnn.mrcnn.utils import Dataset
from src.data import helper
import pandas as pd
import numpy as np


class CityscapeDataset(Dataset):
  def load_dataset(self,dataset_dir,subset):
    # self.image_info = {}
    # self.class_info = {}
    # self.data_directory = dataset_dir

    classes = pd.read_csv(os.path.join(dataset_dir,'classes.csv'))
    classes = classes.set_index('CLASS_ID').T.to_dict('list')
    classes = dict([(k,classes[k][0]) for k,v in classes.items()])
    
    tmp_mapping = {}
    self.rev_mapping = {}
    for i,class_items in enumerate(classes.items()):
      self.add_class(source='cityscape',
                   class_id = i+1,
                   class_name = class_items[1])
      tmp_mapping[class_items[0]] = i+1
      self.rev_mapping[i+1]  = class_items[0]
    
    labels = pd.read_csv(os.path.join(dataset_dir, 'labels_obs.csv'), index_col=None)
    labels = labels.set_index('ID').T.to_dict('list')
    for id in labels:
      labs = labels[id][0]
      obs = labels[id][1]
      labels[id] =  (np.array([tmp_mapping[int(i)] for i in labs.split(' ')[:-1]]),np.array([int(i) for i in obs.split(' ')[:-1]]))
      
      
    # self.class_names = [d['name'] for d in self.class_info.values()]
    
    # iterating to get the image ids
    if subset == 'validation':
      subset = 'val'

    _path = os.path.join(os.path.dirname(dataset_dir), subset+'.csv')
    SUBSET_IDs = pd.read_csv(_path, index_col=0) 
    SUBSET_IDs = [i.split('.')[0] for i in SUBSET_IDs['0'].tolist()]

    for ID in SUBSET_IDs:
    # for i in range(len(data)):
      img_path = os.path.join(dataset_dir, f'images/{ID}.png')
      img_labels = labels[ID][0]
      obs_labels = labels[ID][1]
      self.add_image(source='cityscape',
                     image_id = ID,
                     path = img_path,
                     labels = img_labels.astype(np.int32),
                     obstruction = obs_labels.astype(np.int32))
      
  def load_mask(self,id):
    # if not isinstance(id, str):
    #   id = self.image_info[id]

    img_info = self.image_info[id]
    img_name = img_info['id']
    mask_path = os.path.join(os.path.dirname(os.path.dirname(img_info['path']))
                        ,'segmentations',f'{img_name}.json')

    masks = helper.create_mask(mask_path, img_info['labels'], tmp_mapping=self.rev_mapping).astype(bool)
    return masks, img_info['labels'], img_info['obstruction']
  
  def image_reference(self, id):
    info = self.image_info[id]
    if info["source"] == "cityscape":
        return info["id"]
    else:
        super(self.__class__, self).image_reference(id)

