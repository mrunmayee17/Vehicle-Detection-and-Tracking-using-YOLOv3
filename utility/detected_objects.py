import numpy as np
import cv2
from utility.counting_vehicle import counting_vehicle
from utility.tracker import EuclideanDistTracker

tracker = EuclideanDistTracker()



classesFile = "coco.names"
classNames = open('/Users/mrunmayeerane/Desktop/progress/computervisionbasics/Vehicle tracking/vehicle-detection-classification-opencv/coco_class_index/coco.names').read().strip().split('\n')
print(classNames)
print(len(classNames))

np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')

# Detection confidence threshold
confThreshold =0.2
nmsThreshold= 0.2

# Indices for required detection classes
required_class_index = [2, 3, 5, 7]

detected_classNames = []

# Function for finding the detected objects from the network output
def find_detected_objects(outputs,img, up_list, down_list, temp_up_list, temp_down_list):
    global detected_classNames 
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    # print(classId)
                    w,h = int(det[2]*width) , int(det[3]*height)
                    x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                    boxes.append([x,y,w,h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

 
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    # print(classIds)
    for i in indices.flatten():
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        # print(x,y,w,h)

        color = [int(c) for c in colors[classIds[i]]]
        name = classNames[classIds[i]]
        detected_classNames.append(name)
        # Draw classname and confidence score 
        cv2.putText(img,f'{name.upper()} {int(confidence_scores[i]*100)}%',
                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw bounding rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        detection.append([x, y, w, h, required_class_index.index(classIds[i])])

    # Update the tracker for each object
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        counting_vehicle(box_id, img,up_list,down_list,temp_up_list,temp_down_list)