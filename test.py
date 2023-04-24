import cv2
import pysift
image = cv2.imread('box_in_scene.png', 0)




keypoints, descriptors = pysift.computeKeypointsAndDescriptors(image)
print(len(keypoints))