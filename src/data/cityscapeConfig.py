from models.maskrcnn.mrcnn.config import Config


class CityscapeConfig(Config):
  NAME = 'sidewalks'
  GPU_COUNT = 1

  # 12GB GPU can typical handle 2 images (1024 x 1024)
  # ColabPro High Ram has roughly 27GB GPU
  IMAGES_PER_GPU = 2

  # Background + 1 sidewalk class 
  NUM_CLASSES = 1 + 10

  # Number of training steps per epoch
  STEPS_PER_EPOCH = 50

  # Number of validation steps to runa tt eh end of 
  # every epoch,larger number improves better accuracy
  # but slows down training
  VALIDATION_STEPS = 5
  USE_MINI_MASK = False

  # ROIs belwo this threshold are skipped
  DETECTION_MIN_CONFIDENCE = .7   
  DETECTION_NMS_THRESHOLD = 0.3
  LEARNING_RATE =0.0001
  BACKBONE = 'resnet101'
  IMAGE_MAX_DIM = 512
  IMAGE_MIN_DIM = 512

  LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.,
        "mrcnn_obs_loss": 1.
    }
  
 