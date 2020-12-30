import cv2
import numpy as np

img=np.zeros((100,100,3))
img+=255
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, str(3.12),(50,50),font,0.7,(40,240,0),2)
cv2.imwrite('test.jpg',img)