import cv2
import numpy as np
import os

os.chdir(r'C:\workspace\practiceProject\object_detection\03. OpenCV Practice\img_pro')

src_img = cv2.imread('poker.jpg')
point_list = []

COLOR = (255, 0, 255)
THICKNESS = 3
drawing = False

def mouse_handler(event, x, y, flags, param):
    global drawing
    dst_img = src_img.copy()
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        point_list.append((x, y)) # save coords
    
    if drawing:
        prev_point = None # start of line
        for point in point_list:
            cv2.circle(dst_img, point, 15, COLOR, cv2.FILLED)
            if prev_point:
                cv2.line(dst_img, prev_point, point, COLOR, THICKNESS, cv2.LINE_AA)
            prev_point = point
        
        next_point = (x, y)
        
        if len(point_list) == 4:
            show_result()
            next_point = point_list[0]
        
        cv2.line(dst_img, next_point, point, COLOR, THICKNESS, cv2.LINE_AA)
    
    cv2.imshow('img', dst_img)

def show_result():
    w, h = 530, 710
    src = np.float32(point_list)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src, dst)
    result = cv2.warpPerspective(src_img, matrix, (w, h))
    
    cv2.imshow('result', result)

cv2.namedWindow('img') # window handler for mouse event
cv2.setMouseCallback('img', mouse_handler)

cv2.imshow('img', src_img)
cv2.waitKey(0)
cv2.destroyAllWindows()