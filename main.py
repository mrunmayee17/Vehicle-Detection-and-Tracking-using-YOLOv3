import cv2
import numpy as np

from utility.detected_objects import find_detected_objects


# Initialize the videocapture object
cap = cv2.VideoCapture('/Users/mrunmayeerane/Desktop/progress/computervisionbasics/Vehicle tracking/vehicle-detection-classification-opencv/video_file/video.mp4')

input_size = 320
font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2

# Model Config
modelConfiguration = '/Users/mrunmayeerane/Desktop/progress/computervisionbasics/Vehicle tracking/vehicle-detection-classification-opencv/model_config/yolov3-320.cfg'
modelWeigheights = '/Users/mrunmayeerane/Desktop/progress/computervisionbasics/Vehicle tracking/vehicle-detection-classification-opencv/model_config/yolov3-320.weights'

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

# Enabling GPU processing
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]

def real_time():
    while True:
        success, img = cap.read()
        img = cv2.resize(img,(0,0),None,0.5,0.5)
        ih, iw, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

        # Setting the input of the network
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
        # Feeding data to the network
        outputs = net.forward(outputNames)
    
        # Find the objects from the network output
        find_detected_objects(outputs,img, up_list, down_list, temp_up_list, temp_down_list)

        # Drawing cross measuring line 

        # cv2.line(img, (0, middle_line_position), (iw, middle_line_position), (255, 0, 255), 2)
        # cv2.line(img, (0, up_line_position), (iw, up_line_position), (0, 0, 255), 2)
        # cv2.line(img, (0, down_line_position), (iw, down_line_position), (0, 0, 255), 2)

        # Draw counting texts in the frame
        cv2.putText(img, "Up", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Down", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Car:        "+str(up_list[0])+"     "+ str(down_list[0]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Car:        "+str(up_list[0])+"     ", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Motorbike:  "+str(up_list[1])+"     "+ str(down_list[1]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Bus:        "+str(up_list[2])+"     "+ str(down_list[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Truck:      "+str(up_list[3])+"     "+ str(down_list[3]), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

        # Show the frames
        cv2.imshow('Output', img)

        if cv2.waitKey(1) == ord('q'):
            break

    
    # destroying all active windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    real_time()
