# USAGE
# python simple_cat_dog_test.py /path/to/image_file
from keras.models import load_model
import cv2
import numpy as np
import sys

#loads a pre-trained model for cat-dog classification
model = load_model('/home/devanandt/Documents/ML_HOBBY/kat_shriya_face_CNN_64_64_new.h5')
#model = load_model('/home/devanandt/Documents/ML_HOBBY/kat_shriya_face_32_32_new.h5')
image_test = cv2.imread(sys.argv[1])

im = cv2.resize(image_test, (64,64))
pr = model.predict_classes(np.array([im]))

if pr==[1]:
    print "Shriya Saran"
else:
    print "Katrina Kaif"

