import cv2
import argparse

from time import time
from sklearn import svm
from sklearn.externals import joblib

# add argument parser for images
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--path', help='path to train folder')
ap.add_argument('-q', '--quantity', help='quantity of images in folder')
args = vars(ap.parse_args())

# declare list of data
data = []
label = []

# declare image counter
counter = 1

#start timer
start_time = time()
print ('Start training SVM')

# process image
while counter <= int(args['quantity']):
    image = cv2.imread('{}{}.JPG'.format(args['path'],counter))
    image = cv2.resize(image, (80, 50), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = image.flatten()

    data.append(image)
    label.append(counter)

    counter += 1

# train model
model = svm.SVC()
model.fit(data, label)

# save model
joblib.dump(model, "model.pkl", compress=3)

#end timer
end_time = time() - start_time
print ('Training ended in {} seconds'.format(round(end_time, 2)))