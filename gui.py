from Tkinter import *
from PIL import Image
from PIL import ImageTk
import tkFileDialog
import cv2
import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

def enhance():
	global panelB, gray
	if panelB is not None:
 		panelB.grid_forget()
	dst = cv2.fastNlMeansDenoising(gray,None,5,6,19)
	clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10,10))
	dst = clahe.apply(dst)
	edged = Image.fromarray(dst)
	edged = ImageTk.PhotoImage(edged)

	if panelB is None:
		panelB = Label(image=edged)
		panelB.image = edged
		panelB.pack(side="right", padx=10, pady=10)
	else:
		panelB.pack_forget()
		panelB = Label(image=edged)
		panelB.image = edged
		panelB.pack(side="right", padx=10, pady=10)


def fillHoles(mask):
	maskFloodfill = mask.copy()
	h, w = maskFloodfill.shape[:2]
	maskTemp = np.zeros((h+2, w+2), np.uint8)
	cv2.floodFill(maskFloodfill, maskTemp, (0, 0), 255)
	mask2 = cv2.bitwise_not(maskFloodfill)
	return mask2 | mask

def redeye():
	global panelB, image, gray
	if panelB is not None:
 		panelB.grid_forget()
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
	img_original = image.copy()
	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	if not np.any(faces):
		print "empty"
		eyes = eye_cascade.detectMultiScale(gray)
		for (ex,ey,ew,eh) in eyes:
			roi_eye = image[ey:ey+eh, ex:ex+ew]
			# cv2.imshow('img2', roi_eye)
			# cv2.waitKey(0)
			b = roi_eye[:, :, 0]
			g = roi_eye[:, :, 1]
			r = roi_eye[:, :, 2]
			bg = cv2.add(b,g)
			mask = (r>100)& (r>bg)
			mask = mask.astype(np.uint8)*255
			mask = fillHoles(mask)
			mask = cv2.dilate(mask, None, anchor=(-1, -1), iterations=1, borderType=1, borderValue=1)
			mean = bg / 2
			mask = mask.astype(np.bool)[:, :, np.newaxis]
			mean = mean[:, :, np.newaxis]
			eyeOut = roi_eye.copy()
			np.copyto(eyeOut, mean, where=mask)
			image[ey:ey+eh, ex:ex+ew] = eyeOut
			# cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

	else: 
		for (x,y,w,h) in faces:
			# cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = image[y:y+h, x:x+w]
			
			eyes = eye_cascade.detectMultiScale(roi_gray)
			for (ex,ey,ew,eh) in eyes:
				roi_eye = roi_color[ey:ey+eh, ex:ex+ew]
				# cv2.imshow('img2', roi_eye)
				# cv2.waitKey(0)
				b = roi_eye[:, :, 0]
				g = roi_eye[:, :, 1]
				r = roi_eye[:, :, 2]
				bg = cv2.add(b,g)
				mask = (r>100)& (r>bg)
				mask = mask.astype(np.uint8)*255
				mask = fillHoles(mask)
				mask = cv2.dilate(mask, None, anchor=(-1, -1), iterations=1, borderType=1, borderValue=1)
				mean = bg / 2
				mask = mask.astype(np.bool)[:, :, np.newaxis]
				mean = mean[:, :, np.newaxis]
				eyeOut = roi_eye.copy()
				np.copyto(eyeOut, mean, where=mask)
				roi_color[ey:ey+eh, ex:ex+ew] = eyeOut
	image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	edged = Image.fromarray(image)
	edged = ImageTk.PhotoImage(edged)
	if panelB is None:
		panelB = Label(image=edged)
		panelB.image = edged
		panelB.pack(side="right", padx=10, pady=10)
	else:
		panelB.pack_forget()
		panelB = Label(image=edged)
		panelB.image = edged
		panelB.pack(side="right", padx=10, pady=10)


def select_image():
	# grab a reference to the image panels
	global panelA, panelB, image, gray
	image = None
	gray = None
	if panelA is not None:
 		panelA.grid_forget()
	# open a file chooser dialog and allow the user to select an input
	# image
	path = tkFileDialog.askopenfilename()
	if len(path) > 0:
		image = cv2.imread(path)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# edged = enhance(image);

		# convert the images to PIL format...
		image1 = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
		# edged = Image.fromarray(edged)

		# ...and then to ImageTk format
		image1 = ImageTk.PhotoImage(image1)
		# edged = ImageTk.PhotoImage(edged)

		if panelA is None: 
		# the first panel will store our original image
			panelA = Label(image=image1)
			panelA.image = image1
			panelA.pack(side="left", padx=10, pady=10)

			# while the second panel will store the edge map
			# panelB = Label(image=edged)
			# panelB.image = edged
			# panelB.pack(side="right", padx=10, pady=10)

		# otherwise, update the image panels
		else:
			# update the pannels
			panelA.configure(image=image1)
			# panelB.configure(image=edged)
			panelA.image = image1
			panelB.grid_forget()
			panelB.pack_forget()
			# panelB.image = edged

root = Tk()
panelA = None
panelB = None
image = None
gray = None
 
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
# or panelB is None:

btn = Button(root, text="Enhance", command=enhance)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn = Button(root, text="Correct Red Eye", command=redeye)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn = Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
 
# kick off the GUI
root.mainloop()