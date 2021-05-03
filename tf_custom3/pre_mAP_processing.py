# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:10:27 2021

@author: Steve
"""

import os
import pathlib
os.chdir('models/research/')
os.getcwd()

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile


from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


#Solves the CUDNN error issue
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  model = tf.saved_model.load(str(model_dir))

  return model

os.chdir('../..')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
category_index


#Load our trained model
detection_model = tf.saved_model.load('./my_trained_model/saved_model')

#print(detection_model.signatures['serving_default'].inputs)

#detection_model.signatures['serving_default'].output_dtypes

#detection_model.signatures['serving_default'].output_shapes

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict


def pred_bb_location(model, image_path):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = np.array(Image.open(image_path))

  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  res = vis_util.visualize_boxes_and_labels_on_image_array2(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8,
      category_index = category_index)
  
  im_height, im_width, _ = image_np.shape
  #Scale coordinates back to standard values
  for detection in res:
      detection[0], detection[1], detection[2], detection[3] = (detection[0] * im_width, 
                                    detection[1] * im_height, detection[2] * im_width, detection[3] * im_height)
  
  return res


# =============================================================================
# Function to extract image path name and ground truth bounding box locations from the images
# Ensure to update class_names.txt with the relivent classes
# =============================================================================
print(os.getcwd())
from XML_to_YOLOv3 import run_XML_to_YOLOv3
test_annot_path = "dataset_test.txt"
def extract_annot_data(test_annot_path):
    # Convert XML to YOLOv3
    run_XML_to_YOLOv3()
    
    final_annotations = []
    with open(test_annot_path, 'r') as f:
        txt = f.readlines()
        annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        
    for annotation in annotations:
        # fully parse annotations
        line = annotation.split()
        #print(line[-2:])
        image_path, index = "", 1
        for i, one_line in enumerate(line):
            if not one_line.replace(",","").isnumeric():
                if image_path != "": image_path += " "
                image_path += one_line
            else:
                index = i
                break
        final_annotations.append([image_path, line[index:]])    
    
    return final_annotations

#Extracts the ground truth bb locations from annotations
def gt_bb_location(annotations):
    gt_locations = []
    
    #Runs this block if we are dealing with more than 1 image
    if len(annotations) > 2:
        for i in range(0,len(annotations)):
            temp = annotations[i][1] #seperates only the bb locations for both classes
            image_location = []
            for box in temp:
                temp = box.split(',') #Splits each coordatate
                temp = [int(val) for val in temp] #Conversts from string to integer 
                image_location.append(np.array(temp)) #Appends as array in order to run calculations later
            gt_locations.append(image_location)  
            
    #Runs this block if only single image is given    
    else:    
        for box in annotations[1]:
            box = box.split(',')
            box = [int(val) for val in box]
            gt_locations.append(np.array(box))
    
    #Run only if output giving 2 instead of 1 (temp fix)
    for x in gt_locations:
        if x[0][-1] ==2:
            x[0][-1]=1
    
    return gt_locations


#Extract predicted bounding boxe locations for each test image
def pred_bb_location_annotations(annotations):
    tf_pred_data = []
    for test_image in annotations:
      #print(test_image[0])
      tf_pred_data.append(pred_bb_location(detection_model, test_image[0]))
      
    return tf_pred_data
  
    
annotations = extract_annot_data(test_annot_path)  

gt_boxes = gt_bb_location(annotations)
pred_boxes = pred_bb_location_annotations(annotations) 





                    