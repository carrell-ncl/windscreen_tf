# windscreen_tf

Versions:
Tensorflow 2.3.0


Custom object detector using Tensorflow API to detect 2 classes 1. person using mobile phone when driving, 2. vehicle licence plate

(1)
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

(2)
Check images or video using detection_custom.py

(3)
Calculate model accuracy by running get_AP_mAP.py. Ensure the paths are correct.