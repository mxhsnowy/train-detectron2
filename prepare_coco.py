from detectron2.data.datasets import register_coco_instances
import torch
import cv2
import numpy as np

#* Assume the data is in coco format
def get_data(imgFol, jsonPath, datasetName):
    return register_coco_instances(datasetName, {}, jsonPath, imgFol)

