# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 17:32:04 2021

@author: Steve
"""

from pre_mAP_processing import extract_annot_data, gt_bb_location, pred_bb_location_annotations, test_annot_path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapz



annotations = extract_annot_data(test_annot_path)  
gt_data = gt_bb_location(annotations)
pred_data = pred_bb_location_annotations(annotations) 


# =============================================================================
# bb_intersection_over_union function copied from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/    
# =============================================================================
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou



#Simple function to count the classes. Used in the get_iou_scores function
def get_class(boxes):
    detection = []
    for det in boxes:
        detection.append(det[-1])
    return detection



def get_detection_results(gt_data, pred_data, class_ID=0):
    results = []
    
    for i in range(len(gt_data)):
    # =============================================================================
    #     Below block of code deals with >1 detection of the same class. Tests to see if it is part of the same detection (IOU >0), if so
    #     it will then append the max of these scores. For any scores below IOU of 0, we class these as a false positive
    # =============================================================================
        if get_class(pred_data[i]).count(class_ID) > get_class(gt_data[i]).count(class_ID) and get_class(pred_data[i]).count(class_ID) > 1:
            temp_iou = []
            for detection in pred_data[i]:
                
                if detection[-1] == class_ID and gt_data[i][0][-1] ==class_ID:
                    iou = bb_intersection_over_union(gt_data[i][0][:4], detection[:4])
                    #If IOU score is zero, we know it is outside the area and is a false positive.
                    if iou == 0:
                        results.append([i+1, detection[4], 'FP'])
                    else:
                        temp_iou.append(iou)
                        #print(iou, i)
            #Appends the max of the IOU scores (that are above zero)
            try:        
                results.append([i+1, detection[4], max(temp_iou)])
            except:
                pass
    # =============================================================================
    #     Below handles images where the length of gt and predictions are the same then checks if their class is the same.
    #     if true, it will calculate IOU, if false it will return false negative
    # =============================================================================
        elif len(gt_data[i]) ==1 and len(pred_data[i]) == 1:
            if gt_data[i][0][-1]==class_ID and pred_data[i][0][-1] ==class_ID:
                iou = bb_intersection_over_union(gt_data[i][0][:4], pred_data[i][0][:4])
                results.append([i+1, pred_data[i][0][4], iou])
            #Below finds where no prediction was made for the selected class and returns false negative
            else:
                if gt_data[i][0][-1]==class_ID and not gt_data[i][0][-1] == pred_data[i][0][-1]:
                    results.append([i+1, pred_data[i][0][4], 'FN'])
    # =============================================================================
    #     Handles the remainder of the images and iterates through the detections to find the matching classes and then 
    #     calculates IOU for them.        
    # =============================================================================
        else:
            for detection in pred_data[i]:
                for gt_box in gt_data[i]:
                    if detection[-1] == class_ID and gt_box[-1] == class_ID:
                        iou = bb_intersection_over_union(gt_box[:4], detection[:4])
                        results.append([i+1, detection[4], iou])
    return results

            
results_phone = get_detection_results(gt_data, pred_data, 0)
results_plate = get_detection_results(gt_data, pred_data, 1)

    


# =============================================================================
# Below function to calculate Average precision. Arguments are class and IOU threshold
# =============================================================================
def calculate_AP(class_ID=0, IOU_threshold=0.4):
    results = get_detection_results(gt_data, pred_data, class_ID)
    detections = []
    for val in results:
        try:
            if val[-1] >= IOU_threshold:
                val[-1]='TP'
                detections.append(val)
            else:
                val[-1]='FP'
                detections.append(val)
        except:
            print('NO IOUs')
            detections.append(val)
    
    #Change to dataframe and create columns    
    df = pd.DataFrame(detections, columns=['Image', 'Score', 'Detection']) 
    df = df.sort_values(by='Score', ascending=False)
    
    #Creates extra columns for TP, FP and FN columns
    df['TP'] = df.apply(lambda x: 1 if x['Detection']=='TP' else 0, axis=1)
    df['FP'] = df.apply(lambda x: 1 if x['Detection']=='FP' else 0, axis=1)
    df['FN'] = df.apply(lambda x: 1 if x['Detection']=='FN' else 0, axis=1)
    # Calculate precision and recall and put results into new columns
    df['Precision'] = df['TP'].cumsum()/(df['TP'].cumsum() + df['FP'].cumsum())
    # =============================================================================
    # For recall we can simply use length of DF to get the total number of possible positive due to there being
    # a max of 1 GT class per image (as mentioned earlier)
    # =============================================================================
    df['Recall'] = df['TP'].cumsum()/len(df) 
    prec =  list(df.Precision)
    #Start X at zero to allow us to calculate area under the line.
    prec.insert(0, 1)  
    rec = list(df.Recall)
    rec.insert(0,0)
    
    #Uncomment below to display recall/precision plot
    plt.plot(rec, prec)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
       
    #Calculates area under the line (AR)
    return trapz(prec, rec)    
            


# =============================================================================
# Calculate mAP using the PASCAL VOC method of IoU threshold of 0.5
# =============================================================================
AP_phone = calculate_AP(0, IOU_threshold=0.8)
AP_plate = calculate_AP(1, IOU_threshold=0.8)



print(f'mAP is : {(AP_phone+AP_plate)/2}')





