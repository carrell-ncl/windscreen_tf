# -*- coding: utf-8 -*-
"""
Created on Wed May 26 16:44:15 2021

@author: Steve
"""

from load_model import *



#image_path   = "mAP-master/input/images-optional/mar1.JPG"
image_directory = "mAP-master/input/images-optional"

dic = ['Phone']

def pred_bb_location(model, image_path, image_name):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(image_path))
    #original_image      = cv2.imread(image_path)
    #original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    #original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

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

    #Save detection results in the required order/format
    myfile = open(f'mAP-master/input/detection-results/{image_name}.txt', 'w')  
    
    try:
        for detection in res:
            
            output = list(detection[:5])
            output.insert(0, output.pop())
            #print(pred_locations[0][5])
            output.insert(0, dic[int(detection[5])])
            output = [str(x) for x in output]
            output = ' '.join(output)
            output +='\n'
            myfile.write(output)
            #print(output2)
            myfile.close()
            
    except:
        myfile.close()
        print(f'No detection found in: {image_name}')
  
#pred_bb_location(detection_model, image_path3, 'TEST')    
  
for filename in os.listdir(image_directory):
    if filename.endswith(".JPG") or filename.endswith(".JPEG") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
        try:
            #detection = detection_results(yolo, filename)
            image_name = Path(os.path.join(image_directory, filename)).stem
            #print('HELLO')
            
            detection = pred_bb_location(detection_model, os.path.join(image_directory, filename), image_name)
        except:
            print('ERROR in ', filename)
print('All detection results now saved!')




