#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.

import logging
# cv2 and helper:
import cv2
import sys
import signal
import json
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from helper.common import *
from helper.video import *
# add facerec to system path
sys.path.append("../..")
# facerec imports
from facerec.model import PredictableModel
from facerec.feature import Fisherfaces
from facerec.distance import EuclideanDistance
from facerec.classifier import NearestNeighbor
from facerec.validation import KFoldCrossValidation
from facerec.serialization import save_model, load_model
# for face detection (you can also use OpenCV2 directly):
from facedet.detector import CascadedDetector

# This system command loads the right drivers for the Raspberry Pi camera
#os.system('sudo modprobe bcm2835-v4l2')

class ExtendedPredictableModel(PredictableModel):
    """ Subclasses the PredictableModel to store some more
        information, so we don't need to pass the dataset
        on each program call...
    """

    def __init__(self, feature, classifier, image_size, subject_names):
        PredictableModel.__init__(self, feature=feature, classifier=classifier)
        self.image_size = image_size
        self.subject_names = subject_names

def get_model(image_size, subject_names):
    """ This method returns the PredictableModel which is used to learn a model
        for possible further usage. If you want to define your own model, this
        is the method to return it from!
    """
    # Define the Fisherfaces Method as Feature Extraction method:
    feature = Fisherfaces()
    # Define a 1-NN classifier with Euclidean Distance:
    classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=1)
    # Return the model as the combination:
    return ExtendedPredictableModel(feature=feature, classifier=classifier, image_size=image_size, subject_names=subject_names)

def read_subject_names(path):
    """Reads the folders of a given directory, which are used to display some
        meaningful name instead of simply displaying a number.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).

    Returns:
        folder_names: The names of the folder, so you can display it in a prediction.
    """
    folder_names = []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            folder_names.append(subdirname)
    return folder_names

def read_images(path, image_size=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes 

    Returns:
        A list [X, y, folder_names]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
            folder_names: The names of the folder, so you can display it in a prediction.
    """
    c = 0
    X = []
    y = []
    folder_names = []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            folder_names.append(subdirname)
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    # resize to given size (if given)
                    if (image_size is not None):
                        im = cv2.resize(im, image_size)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError:
                    print ("I/O error({0}): {1}".format(err.errno, err.strerror))
                except:
                    print ("Unexpected error:", sys.exc_info()[0])
                    raise
            c = c+1
    return [X,y,folder_names]

# taken from https://github.com/paviro/MMM-Facial-Recognition/blob/master/facerecognition/facerecognition.py#L34-L41
def to_node(type, message):
    # convert to json and print (node helper will read from stdout)
    try:
        if not isinstance(message, dict):
            message = {"message": message}
        print(json.dumps({type: message}))
    except Exception:
        pass
    # stdout has to be flushed manually to prevent delays in the node helper communication
    sys.stdout.flush()

class App(object):
    def __init__(self, model, camera_id, cascade_filename):
        signal.signal(signal.SIGINT, self.shutdown)

        self.model = model
        self.detector = CascadedDetector(cascade_fn=cascade_filename, minNeighbors=5, scaleFactor=1.1)
        
        try:
            self.cam = create_capture(camera_id)
        except:
            to_node("error", "Camera '%s' unable to connect." % camera_id)
            sys.exit()

        self.user = None
        self.faces = None
        self.has_changed = {"face_count": False, "user": False}
        to_node("status", {"camera": str(camera_id), "model": str(model), "detector": str(self.detector)})

    def find_faces(self, img):
        faces = self.detector.detect(img)

        self.has_changed['face_count'] = self.faces is None or len(faces) is not len(self.faces)
        self.faces = faces

        return(faces)

    def shutdown(self, signum, stack):
        to_node("status", 'Shutdown -- Cleaning up camera...')
        self.cam.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    def run(self):
        while True:
            ret, frame = self.cam.read()
            # Resize the frame to half the original size for speeding up the detection process:
            img = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)), interpolation = cv2.INTER_CUBIC)
            imgout = img.copy()
            self.find_faces(img)

            if self.has_changed['face_count']:
                to_node("change", {"face_count": len(self.faces)})

            for i,r in enumerate(self.faces):
                x0,y0,x1,y1 = r
                # (1) Get face, (2) Convert to grayscale & (3) resize to image_size:
                face = img[y0:y1, x0:x1]
                face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, self.model.image_size, interpolation = cv2.INTER_CUBIC)
                # Get a prediction from the model:
                prediction = self.model.predict(face)[0]
                confidence = self.model.predict(face)[1]["distances"][0]
                # Draw the face area in image:
                cv2.rectangle(imgout, (x0,y0),(x1,y1),(0,255,0),2)
                # Draw the predicted name (folder name...):
                #print (confidence)

                if confidence < 550 and self.model.subject_names[prediction] is not None:
                    user = self.model.subject_names[prediction]
                else:
                    user = ""

                self.has_changed['user'] = self.user is None or user is not self.user

                if self.has_changed['user']:
                    self.user = user
                    to_node("change", {"user": user, "confidence": confidence})


if __name__ == '__main__':
    from argparse import ArgumentParser
    # model.pkl is a pickled (hopefully trained) PredictableModel, which is
    # used to make predictions. You can learn a model yourself by passing the
    # parameter -d (or --dataset) to learn the model from a given dataset.

    parser = ArgumentParser(description="face recognizer")
    parser.add_argument("model_filename", type=str, default="model.pkl", nargs="?")
    parser.add_argument("-r", "--resize", type=str, dest="size", default="100x100", 
        help="Resizes the given dataset to a given size in format [width]x[height] (default: 100x100).")
    parser.add_argument("-v", "--validate", dest="numfolds", type=int, default=None, 
        help="Performs a k-fold cross validation on the dataset, if given (default: None).")
    parser.add_argument("-t", "--train", dest="dataset", type=str, default=None,
        help="Trains the model on the given dataset.")
    parser.add_argument("-i", "--id", dest="camera_id", type=int, default=0, 
        help="Sets the Camera Id to be used (default: 0).")
    parser.add_argument("-c", "--cascade", dest="cascade_filename", type=str, default="haarcascade_frontalface_alt2.xml",
        help="Sets the path to the Haar Cascade used for the face detection part (default: haarcascade_frontalface_alt2.xml).")
    args = parser.parse_args()

    # This model will be used (or created if the training parameter (-t, --train) exists:

    # Check if the given model exists, if no dataset was passed:
    if (args.dataset is None) and (not os.path.exists(args.model_filename)):
        to_node("error", "No prediction model found at '%s'." % args.model_filename)
        sys.exit()

    # Check if the given (or default) cascade file exists:
    if not os.path.exists(args.cascade_filename):
        to_node("error", "No Cascade File found at '%s'." % args.cascade_filename)
        sys.exit()
    # We are resizing the images to a fixed size, as this is neccessary for some of
    # the algorithms, some algorithms like LBPH don't have this requirement. To 
    # prevent problems from popping up, we resize them with a default value if none
    # was given:
    try:
        image_size = (int(args.size.split("x")[0]), int(args.size.split("x")[1]))
    except:
        to_node("error", "Unable to parse the given image size '%s'. Please pass it in the format [width]x[height]!" % args.size)
        sys.exit()
    # We have got a dataset to learn a new model from:

    model = load_model(args.model_filename)
    to_node("status", "Model '%s' loaded." % args.model_filename)

    # We operate on an ExtendedPredictableModel. Quit the application if this
    # isn't what we expect it to be:
    if not isinstance(model, ExtendedPredictableModel):
        to_node("error", "The given model is not of type '%s'." % "ExtendedPredictableModel")
        sys.exit()
    # Now it's time to finally start the Application! It simply get's the model
    # and the image size the incoming webcam or video images are resized to:
    #print ("Starting application...")
    App(model=model,
        camera_id=args.camera_id,
        cascade_filename=args.cascade_filename).run()
