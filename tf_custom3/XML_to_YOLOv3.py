#================================================================
#
#   File name   : XML_to_YOLOv3.py
#   Author      : PyLessons
#   Created date: 2020-06-04
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : used to convert XML labels to YOLOv3 training labels
#
#================================================================
import xml.etree.ElementTree as ET
import os
import glob

foldername = os.path.basename(os.getcwd())
if foldername == "tools": os.chdir("..")


data_dir = "\images/"
Dataset_names_path = "class_names.txt"
Dataset_test = "dataset_test.txt"
is_subfolder = True

Dataset_names = []
      
def ParseXML(img_folder, file):
    for xml_file in glob.glob(img_folder+'/*.xml'):
        tree=ET.parse(open(xml_file))
        root = tree.getroot()
        print(img_folder)
        image_name = root.find('filename').text
        img_path = img_folder+'/'+image_name
        for i, obj in enumerate(root.iter('object')):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in Dataset_names:
                Dataset_names.append(cls)
            cls_id = Dataset_names.index(cls)
            xmlbox = obj.find('bndbox')
            OBJECT = (str(int(float(xmlbox.find('xmin').text)))+','
                      +str(int(float(xmlbox.find('ymin').text)))+','
                      +str(int(float(xmlbox.find('xmax').text)))+','
                      +str(int(float(xmlbox.find('ymax').text)))+','
                      +str(cls_id))
            img_path += ' '+OBJECT
            
        #print(img_path)
        file.write(img_path+'\n')

def run_XML_to_YOLOv3():
    with open(Dataset_test, "w") as file:
        #print(os.getcwd()+data_dir+'test')
        img_path = os.path.join(os.getcwd()+data_dir+'test')
        ParseXML(img_path, file)

    #print("Dataset_names:", Dataset_names)
    with open(Dataset_names_path, "w") as file:
        for name in Dataset_names:
            file.write(str(name)+'\n')

#run_XML_to_YOLOv3()


