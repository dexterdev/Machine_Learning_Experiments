# USAGE
# python simple_cat_dog_test.py /path/to/image_file
from keras.models import load_model
import cv2
import numpy as np
import sys

#loads a pre-trained model for cat-dog classification
model = load_model('/home/devanandt/Documents/ML_HOBBY/kat_shriya_face_32_32_new.h5')
#face_cascade = cv2.CascadeClassifier('/home/devanandt/Documents/anaconda/pkgs/opencv3-3.1.0-py27_0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
image_test = cv2.imread(sys.argv[1])

#gray = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)
#faces = face_cascade.detectMultiScale(gray, 1.1, 3)
#ind = np.lexsort((faces[:,1],faces[:,2]))
#faces=faces[ind[1],:]
#print faces
#cv2.imwrite( "face.jpg"  ,   image_test[faces[1]:(faces[1]+faces[3]),faces[0]:(faces[0]+faces[2])  ,  :]  )
#image = cv2.imread("./face.jpg")
image=image_test
im = cv2.resize(image, (32,32)).flatten()
pr = model.predict_classes(np.array([im]))

if pr==[1]:
    print "Shriya Saran"
else:
    print "Katrina Kaif"

