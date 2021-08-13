# Identification of Driver Phone Usage Violations using Tensorflow API - benchmarking multiple model architectures

## Hardware/Software
- NVIDIA RTX 2080
- AMD Ryzen 3800X

- Python 3.8
- Tensorflow 2.2.0
- Numpy 1.18.5
- Pandas 1.2.3


Custom object detector using Tensorflow API to detect 2 classes 1. person using mobile phone when driving, 2. vehicle licence plate

![](detection.gif)

## Train
Train model (skip if you do not need to train a new model).
Open tf_custom.ipynb in notebooks or Google Colab
Steps:
- Create an a new file directory in Google Colab then mount drive. Within this have a images folder,
then within that have train and test image. Ensure to have the corresponding .XML files.
- Download Tensorflow 2.3.0
- Git clone the Tensorflow models API
- Run next cells to setup and testing
- Generate the train and test.record files
- Download the chosen model type
- Ensure the .config and label_map.PBTXT file are edited to suit file paths and classes
- Train model
- Output the trained model to .pb file

## Run model
Check images or video using detection_custom.py

## AP and mAP
- Test images saved in mAP-master/input/images-optional
- Annotations (Pascal format) saved in mAP-master/input/ground-truth (file names to be same name as image / file)
- Run 'get_detection_results.py' to create detections files
- Select IOU threshold in main.py - default set to 0.5
- Set CWD to ./mAP-master and run 'main.py'

File structure should look like the following:
```bash
└───tf_custom3
    ├───images
    │   ├───test
    │   └───train
    ├───mAP-master
    │   ├───input
    │   │   ├───detection-results
    │   │   │   └───archive
    │   │   ├───ground-truth
    │   │   │   ├───archive
    │   │   │   └───backup
    │   │   └───images-optional
    │   ├───output
    │   │   ├───classes
    │   │   └───images
    │   │       └───detections_one_by_one
    │   └───scripts
    │       └───extra
    ├───models
    │   ├───***directories cloned from Tensorflow models API
    ├───my_trained_model
    │   ├───checkpoint
    │   └───saved_model
    │       ├───assets
    │       └───variables
    └───__pycache__
```

