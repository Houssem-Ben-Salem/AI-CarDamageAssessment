import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# Function to check and register dataset
def check_and_register(dataset_name, json_file, image_root):
    if dataset_name not in DatasetCatalog:
        register_coco_instances(dataset_name, {}, json_file, image_root)

# Assuming your dataset is in COCO format
# Check and register the datasets if not already registered
check_and_register("my_dataset_train", 
                   "/home/hous/Desktop/LLAVA/CarDD_release/CarDD_COCO/annotations/instances_train2017.json", 
                   "/home/hous/Desktop/LLAVA/CarDD_release/CarDD_COCO/train2017")

check_and_register("my_dataset_val", 
                   "/home/hous/Desktop/LLAVA/CarDD_release/CarDD_COCO/annotations/instances_val2017.json", 
                   "/content/drive/MyDrive/CarDD_release/CarDD_COCO/val2017")

# Getting the metadata
car_damage_metadata = MetadataCatalog.get("my_dataset_train")
def setup_cfg(weights_path):
    """
    Set up and return the Detectron2 config.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # Change as per your model's classes
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # Set the detection threshold
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

def load_model(weights_path):
    """
    Load the Detectron2 model with the specified weights.
    """
    cfg = setup_cfg(weights_path)
    return DefaultPredictor(cfg)


def detect_damage(image, model, metadata):
    print(car_damage_metadata)
    """
    Perform car damage detection and return the image with visualized predictions.
    """
    outputs = model(image)
    v = Visualizer(image[:, :, ::-1],
                   metadata=metadata,
                   scale=1.2,)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1]

