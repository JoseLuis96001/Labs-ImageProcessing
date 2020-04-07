import numpy as np
import cv2 as cv

def click(event, x, y, flags, param):
	global filt_2D,radius


	if event == cv.EVENT_LBUTTONDOWN:
		x_init        = x-radius
		x_fin         = x+radius
		y_init        = y-radius
		y_fin         = y+radius
		filt_2D[y_init:y_fin+1,x_init:x_fin+1,:]=0
		filt_2D[filt_2D.shape[0]-y_fin:filt_2D.shape[0]-y_init
		+1, filt_2D.shape[1]-x_fin:filt_2D.shape[1]-x_init+1,:]=0
		print(x,y)
		cv.circle(cv_magnitude, (x, y), radius, (0, 255, 0), -1)


def modifyRadius(x):
			global radius
			radius = x

image = cv.imread("car.png",0).astype(np.float32)/255

keepProcessing = True

radius  = 1
radius2 = 1

originalName = 'Original Image'
spectrumMagName = 'Spectrum'
filterName = 'Filter'
reconstructedName = 'Reconstructed Image'


cv.namedWindow("c",cv.WINDOW_NORMAL)
cv.resizeWindow("c",650,750)
cv.setMouseCallback('c', click)
cv.createTrackbar('Radius','c',radius,10,modifyRadius)


filt_2D = np.ones((image.shape[0] & -2, image.shape[1] & -2, 2))

fft = cv.dft(image, flags=cv.DFT_COMPLEX_OUTPUT)
fft_shift = np.fft.fftshift(fft,axes=[0,1])
fft_shift*=filt_2D

fft = np.fft.ifftshift(fft_shift, axes=[0,1])

shifted = np.fft.fftshift(fft, axes=[0,1])
magnitude = cv.magnitude(shifted[:,:,0],shifted[:,:,1])
magnitude = np.log(magnitude+1)
cv_magnitude = np.array([])
cv_magnitude = cv.normalize(magnitude,cv_magnitude,0,1,cv.NORM_MINMAX)

reconstructed = cv.idft(fft, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)

#cv.imshow(spectrumMagName,cv_magnitude)

while (keepProcessing):
	
	fft = cv.dft(image, flags=cv.DFT_COMPLEX_OUTPUT)
	fft_shift = np.fft.fftshift(fft,axes=[0,1])
	fft_shift *= filt_2D	

	fft = np.fft.ifftshift(fft_shift, axes=[0,1])

	reconstructed=cv.idft(fft,flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)

	a=np.hstack((image,reconstructed))
	b=np.hstack((cv_magnitude,filt_2D[:,:,0]))
	c=np.vstack((b,a))
	cv.namedWindow("c",cv.WINDOW_NORMAL)
	cv.resizeWindow("c",650,750)
	cv.imshow("c",c)


	key = cv.waitKey(1) & 0xFF
	if key == ord('e'):
		break