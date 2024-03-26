import cv2
import numpy as np
image= cv2.imread("bLoBs.jpg")

params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 100

params.filterByCircularity= True
params.minCircularity = 0.2

params.filterByConvexity= True
params.minConvexity= 0.9

params.filterByInertia= True
params.minInertiaRatio= 0.01

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(image)

blank = np.zeros((1,1))

#drawKeypoints(input_image, key_points, output_image, colour, flag)
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

num=len(keypoints)
text="the blobs: "+str(num)
cv2.putText(blobs,text,(20,550),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(0,0,0),2)

cv2.imshow("result",blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()