import cv2
import numpy as np

img=cv2.imread("bLoBs.jpg",cv2.IMREAD_COLOR)

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray_blurred=cv2.blur(gray,(3,3))

#apply hough transformation function

"""
Detection Method: OpenCV has an advanced implementation, HOUGH_GRADIENT, which uses gradient of the edges instead of filling up the entire 3D accumulator matrix, thereby speeding up the process.
dp: This is the ratio of the resolution of original image to the accumulator matrix.
minDist: This parameter controls the minimum distance between detected circles.
Param1: Canny edge detection requires two parameters — minVal and maxVal. Param1 is the higher threshold of the two. The second one is set as Param1/2.
Param2: This is the accumulator threshold for the candidate detected circles. By increasing this threshold value, we can ensure that only the best circles, corresponding to larger accumulator values, are returned.
minRadius: Minimum circle radius.
maxRadius: Maximum circle radius.
"""


detected_circles = cv2.HoughCircles(gray_blurred,cv2.HOUGH_GRADIENT,1,20, param1 = 50 ,param2= 30,minRadius=1, maxRadius=40)

#ThE ArT PrOcEsS

#DRaWiNg CiRcLeas ArOuNd CiRcle

if detected_circles is not None:
    detected_circles = np.uint16(np.around(detected_circles))

    for i in detected_circles[0,:]:
        a,b,r= i [0],i[1] , i[2]
        cv2.circle(img, (a,b),r,(0,225,0),2)
        cv2.circle(img,(a,b),1,(0,0,225),3)
    cv2.imshow("corcle",img)
    cv2.waitKey(0)

cv2.destroyAllWindows()