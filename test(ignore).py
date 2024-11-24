import cv2
import os
print("Prototxt path:", os.path.abspath('face_detection_model/deploy.prototxt'))
print("Caffemodel path:", os.path.abspath('face_detection_model/res10_300x300_ssd_iter_140000.caffemodel'))
print("Image path:", os.path.abspath('image.jpg'))
detector = cv2.dnn.readNetFromCaffe('face_detection_model/deploy.prototxt', 'face_detection_model/res10_300x300_ssd_iter_140000.caffemodel')

image = cv2.imread('image.jpg') 
if image is None:
    print("Error: Image not found.")
else:
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)

    detections = detector.forward()

    print(detections)