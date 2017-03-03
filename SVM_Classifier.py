import sys
import cv2
import argparse
import warnings

from time import time
from docx import Document
from sklearn.externals import joblib

reload(sys)
sys.setdefaultencoding('utf-8')

# ignore version warnings
def warn(*args, **kwargs):
    pass

warnings.warn = warn

# add argument parser for images
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help='path to .jpg file')
args = vars(ap.parse_args())

#start timer
start_time = time()
print ('Start recognition')

# read image and model
image = cv2.imread(args['image'])
model = joblib.load("model.pkl")

# process image
image = cv2.resize(image, (80, 50), interpolation=cv2.INTER_AREA)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
threshold, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
image = image.flatten()

# predict class
result = model.predict(image)

# print the results
document = Document('txt/{}.docx'.format(result[0]))
for para in document.paragraphs:
    print (para.text)

document.save('output.docx')

#end timer
end_time = time() - start_time
print ('Recognition ended in {} seconds'.format(round(end_time, 2)))