""" 
     
     This script can be used to capture images for an individual to be 
     added to a face library for face learning.

"""
import os
import sys
import select

import cv2
import numpy as np

# This system command loads the right drivers for the Raspberry Pi camera
#os.system('sudo modprobe bcm2835-v4l2')


# Directories which contain the positive and negative training image data.
IMAGE_DIR = './orl_faces'

# Size (in pixels) to resize images for training and prediction.
# Don't change this unless you also change the size of the training images.
FACE_WIDTH  = 92
FACE_HEIGHT = 112

HAAR_FACES         = 'haarcascade_frontalface_alt2.xml'
HAAR_EYES          = 'haarcascade_eye.xml'
HAAR_SCALE_FACTOR  = 1.3
HAAR_MIN_NEIGHBORS = 4
HAAR_MIN_SIZE      = (30, 30)

face_cascade = cv2.CascadeClassifier(HAAR_FACES)
eye_cascade = cv2.CascadeClassifier(HAAR_EYES)


def is_letter_input(letter):
	# Utility function to check if a specific character is available on stdin.
	# Comparison is case insensitive.
	if select.select([sys.stdin,],[],[],0.0)[0]:
		input_char = sys.stdin.read(1)
		return input_char.lower() == letter.lower()
	return False

def enter_pressed():
	# Utility function to check if a specific character is available on stdin.
	# Comparison is case insensitive.
	if select.select([sys.stdin,],[],[],0.0)[0]:
		input_char = sys.stdin.read(1)
		print(input_char)
		return input_char.lower() == "\n"
	return False

def show_capture_window(camera, message):
	image_captured = False
	return_image = None
	faces = []	
	eyes = []
	
	if (camera.isOpened() == 0):
		print("Webcam cannot open!\n")
		return None

	print (message)
	print ("press enter to capture image")
	while  not image_captured:

		# Capture frame-by-frame
		ret, image = camera.read()

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		faces = face_cascade.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(30, 30)
			#flags=cv2.cv.CV_HAAR_SCALE_IMAGE
			)

		# Draw a rectangle around the faces
		for (x, y, w, h) in faces:
			cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = image[y:y+h, x:x+w]
			# Draw a rectangle around the faces
			eyes = eye_cascade.detectMultiScale(roi_gray)
			for (ex,ey,ew,eh) in eyes:
				cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

		cv2.putText(image,message,(25,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv2.LINE_AA)
		# Display the resulting frame
		cv2.imshow('Video', image)
		cv2.moveWindow('Video',100,100)
		cv2.waitKey(10)	
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		if enter_pressed():
			if len(eyes) == 2:
				ret, image = camera.read()
				return_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
				image_captured = True
			else:
				print ("Unable to detect a proper face or too many faces detected: try again")
				print ("press c to capture image")
	# When everything is done, release the capture
	cv2.destroyAllWindows()
	return return_image

def detect_single(image):
	"""Return bounds (x, y, width, height) of detected face in grayscale image.
	   If no face or more than one face are detected, None is returned.
	"""
	faces = face_cascade.detectMultiScale(image, 
				scaleFactor=HAAR_SCALE_FACTOR, 
				minNeighbors=HAAR_MIN_NEIGHBORS, 
				minSize=HAAR_MIN_SIZE, 
				flags=cv2.CASCADE_SCALE_IMAGE)
	if len(faces) != 1:
		return None
	return faces[0]

def detect_eyes(image, face):
	"""Return bounds (x, y, width, height) of detected eyes in grayscale image.
	   If no eyes or more than two eyes are detected, None is returned.
	"""
	eyes = []
	x, y, w, h = face
	cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
	roi_gray = gray[y:y+h, x:x+w]
	roi_color = image[y:y+h, x:x+w]
	# Draw a rectangle around the faces
	eyes = eye_cascade.detectMultiScale(roi_gray)

	if len(eyes) != 2:
		return None
	return eyes	


def crop(image, x, y, w, h):
	"""Crop box defined by x, y (upper left corner) and w, h (width and height)
	to an image with the same aspect ratio as the face training data.  Might
	return a smaller crop if the box is near the edge of the image.
	"""
	crop_height = int((FACE_HEIGHT / float(FACE_WIDTH)) * w)
	midy = y + h/2
	y1 = max(0, midy-crop_height/2)
	y2 = min(image.shape[0]-1, midy+crop_height/2)
	return image[y1:y2, x:x+w]

def resize(image):
	"""Resize a face image to the proper size for training and detection.
	"""
	return cv2.resize(image, 
		(FACE_WIDTH, FACE_HEIGHT), 
		interpolation=cv2.INTER_LANCZOS4)


def img_message(img_num):
	switcher = {
		0: "normal face",
		1: "smile",
		2: "sad face",
		3: "suprised",
		4: "look slightly to the right",
		5: "look slightly to the left",
		6: "look slightly up",
		7: "look slightly down",
		8: "lighting from the left",
		9: "lighting from the right",
		10: "eyes closed"
	}
	return switcher.get(img_num, "nothing")

def img_name(img_num):
	switcher = {
		0: "normal",
		1: "happy",
		2: "sad",
		3: "suprised",
		4: "looking_right",
		5: "looking_left",
		6: "looking_up",
		7: "looking_down",
		8: "left_light",
		9: "right_light",
		10: "eyes_closed"
	}
	return switcher.get(img_num, "nothing")

if __name__ == '__main__':

	#ubuntu camera
	camera = cv2.VideoCapture(0)
	ret = camera.set(3,320)
	ret = camera.set(4,240)
	print ('This app will capture several images to learn your face.')
	name = input('Please enter your name:')
	print ('Press Ctrl-C to quit.')	
	# Create the directory for positive training images if it doesn't exist.
	img_dir = IMAGE_DIR + '/' + name
	if not os.path.exists(img_dir):
		os.makedirs(img_dir)

	img_num = 0
	while img_num < 11:
		# Show the capture window
		image = show_capture_window(camera, img_message(img_num))
		# Get coordinates of single face in captured image.
		result = detect_single(image)

		if result is None:
			print ('Could not detect single face!  Check the image in capture.pgm' \
				  ' to see what was captured and try again with only one face visible.')
			continue

		x, y, w, h = result
		# Crop image as close as possible to desired face aspect ratio.
		# Might be smaller if face is near edge of image.
		img_crop = crop(image, x, y, w, h)
		# Save image to file.
		filename = os.path.join(img_dir, '%02d_%s_%s.pgm' % (img_num, name, img_name(img_num)))
		cv2.imwrite(filename, img_crop)
		print ('Found face and wrote training image' + filename)
		img_num += 1

