# -*- coding: utf-8 -*-
"""
Created on Tue May  4 18:18:19 2021

@author: Steve
"""

import cv2
from load_model import detection_model, run_inference_for_single_image, show_inference
import time
import tensorflow as tf

#Solves the CUDNN error issue
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

video_path = 'images/street2.avi'
image_path = 'images/test_phone1.jpg'

#Make inference on video
def detect_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fps_list = []
    times = []
    
    if video_path:
        vid = cv2.VideoCapture(video_path) # detect on video
    else:
        vid = cv2.VideoCapture(0) # detect from webcam
    
    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4
    
    while True:
        _,img = cap.read()
        
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        t1 = time.time()
        final_img = show_inference(detection_model,img)
        t2 = time.time()
        final_img = cv2.cvtColor(final_img,cv2.COLOR_RGB2BGR)
        
        times.append(t2-t1)
        times = times[-20:]
        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        fps_list.append(fps)
        final_img = cv2.putText(final_img, "FPS: {:.1f}".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        cv2.imshow('img',final_img)
        
        if output_path != '': out.write(final_img)
        
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
    print(sum(fps_list)/len(fps_list))
    cv2.destroyAllWindows()
    
    
#Make inference on video
def detect_video2(video_path):
    cap = cv2.VideoCapture(video_path)
    fps_list = []
    times = []
    while True:
        _,img = cap.read()
        
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        t1 = time.time()
        final_img = show_inference(detection_model,img)
        t2 = time.time()
        final_img = cv2.cvtColor(final_img,cv2.COLOR_RGB2BGR)
        
        times.append(t2-t1)
        times = times[-20:]
        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        fps_list.append(fps)
        final_img = cv2.putText(final_img, "FPS: {:.1f}".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        cv2.imshow('img',final_img)
        
        
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
    print(sum(fps_list)/len(fps_list))
    cv2.destroyAllWindows()



#Make inference on single image
def detect_image(image_path):
    img = cv2.imread(image_path)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = show_inference(detection_model,img)             
    cv2.imshow('img',img)
    
    cv2.waitKey(0)
    # To close the window after the required kill value was provided
    cv2.destroyAllWindows()
    

detect_video(video_path, 'detection.mp4')
detect_image(image_path)
