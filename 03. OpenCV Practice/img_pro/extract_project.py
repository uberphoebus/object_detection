# contour area
import cv2

img = cv2.imread('./card.png')
target_img = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, otsu = cv2.threshold(gray, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# contours info, hierarchy: img, mode, method
contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# draw: img, contours info, contours idx, color, thickness
COLOR = (0, 200, 0)
for cnt in contours:
    if cv2.contourArea(cnt) > 25000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(target_img, 
                          (x, y), # left top
                          (x + w, y + h), # right bottom
                          COLOR, 2)
            
            crop = img[y:]


cv2.imshow('target_img', target_img)
cv2.waitKey(0)
cv2.destroyAllWindows()