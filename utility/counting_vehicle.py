import cv2
from utility.find_center import find_center

# Middle cross line position
middle_line = 520
upper_line = middle_line - 100
down_line = middle_line + 100

def counting_vehicle(box_id, img, up_list, down_list, temp_up_list, temp_down_list):

    x, y, w, h, id, index = box_id

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center
    
    if upper_line  < iy < middle_line:
        if id not in temp_up_list:
            temp_up_list.append(id)
    elif middle_line < iy < down_line:
        if id not in temp_down_list:
            temp_down_list.append(id)
    elif iy < upper_line:
        if id in temp_down_list:
            temp_down_list.remove(id)
            up_list[index] = up_list[index] + 1
    elif iy > down_line:
        if id in temp_up_list:
            temp_up_list.remove(id)
            down_list[index] = down_list[index] + 1

    # Draw circle in the middle of the rectangle
    cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here
    print(up_list, down_list)
