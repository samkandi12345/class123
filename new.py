from logging import exception
import cv2
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time

X,y = fetch_openml("mnist_784",version=1,return_X_y = True)
print(pd.Series(y).value_counts())
classes = ["0","1","2","3","4","5","6","7","8","9"]
nclasses = len(classes)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=40)
x_train_scaled = x_train/255
x_test_scaled = x_test/255
clf = LogisticRegression(solver = "saga",multi_class = "multinomials").fit(x_train_scaled,y_train)
y_predict = clf.predict(x_test_scaled)
accuracy = accuracy_score(y_test,y_predict)
print("accuracy: ",accuracy)

image_bw_resized_inverted_scale = np.asarray(image_bw_resized_inverted_scale)/np.maximumpixel
testsample = clf.predict(testsample)
print("test prediction")
cv2.imshow("frame",gray)

if cv2.waitKey(1) & 0xFF == ord("q"):
    continue

Cap = cv2.VideoCapture(0)
while(Cap.isopenend()):
    ret,img = Cap.read()
    if not ret:
        break
    image_bw_resized_inverted_scale = PIL.ImageOps.invert(image_bw_resized_inverted_scale)
    pixel_filter = 20
    minimum_pixel = np.percentile(image_bw_resized_inverted_scale)
    image_bw_resized_inverted_scale = np.clip(image_bw_resized_inverted_scale,0,255)
    maximum_pixel = np.max(image_bw_resized_inverted_scale)
    testsample = np.array(image_bw_resized_inverted_scale).reshape(1,784)
    test_predict = clf.predict(testsample)
    print(test_predict)
