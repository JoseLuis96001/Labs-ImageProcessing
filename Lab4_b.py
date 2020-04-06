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
		cv.imshow(spectrumMagName,cv_magnitude)

'''
def clickFilter(event, x, y, flags, param):
	global filt_2D,radius2


	if event == cv.EVENT_LBUTTONDOWN:
		x_init        = x-radius2
		x_fin         = x+radius2
		y_init        = y-radius2
		y_fin         = y+radius2
		filt_2D[y_init:y_fin+1,x_init:x_fin+1,:]=1
		filt_2D[filt_2D.shape[0]-y_fin:filt_2D.shape[0]-y_init
		+1, filt_2D.shape[1]-x_fin:filt_2D.shape[1]-x_init+1,:]=1
		print('filtro: ',x,y)
'''		

def modifyRadius(x):
			global radius
			radius = x
'''
def modifyRadius2(x):
			global radius2
			radius2 = x
'''


image = cv.imread("car.png",0).astype(np.float32)/255

keepProcessing = True

radius  = 1
radius2 = 1

originalName = 'Original Image'
spectrumMagName = 'Spectrum'
filterName = 'Filter'
resultName = 'Spectrum x Filter'
reconstructedName = 'Reconstructed Image'

cv.namedWindow(originalName,cv.WINDOW_NORMAL)
cv.resizeWindow(originalName,450,450)

cv.namedWindow(spectrumMagName, cv.WINDOW_NORMAL)
cv.resizeWindow(spectrumMagName, 450,450)
cv.setMouseCallback(spectrumMagName, click)
cv.createTrackbar('Radius',spectrumMagName,radius,10,modifyRadius)


cv.namedWindow(filterName,cv.WINDOW_NORMAL)
cv.resizeWindow(filterName, 450,450)
'''
cv.setMouseCallback(filterName, clickFilter)
cv.createTrackbar("Radius",filterName,radius2,20,modifyRadius2)
'''

cv.namedWindow(resultName,cv.WINDOW_NORMAL)
cv.resizeWindow(resultName, 450,450)

cv.namedWindow(reconstructedName, cv.WINDOW_NORMAL)
cv.resizeWindow(reconstructedName, 450,450)


cv.imshow(originalName,image)

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

cv.imshow(spectrumMagName,cv_magnitude)
#cv.imshow(filterName,filt_2D[:,:,0])
cv.imshow(resultName,fft_shift[:,:,0])
#cv.imshow(reconstructedName,reconstructed)


'''
d1=cv_magnitude
print(cv_magnitude.shape)
print(filt_2D[:,:,0].shape)
print(fft_shift[:,:,0].shape)
print(reconstructed.shape)
#d2=
#d3=
#d4=

f1=cv.hconcat([cv_magnitude,filt_2D[:,:,0]])
#f2=cv.hconcat([fft_shift[:,:,0],reconstructed])
#cv.imdecode('f1',f1)
#cv.imdecode('f2',f2)
'''
while (keepProcessing):
	
	fft = cv.dft(image, flags=cv.DFT_COMPLEX_OUTPUT)
	fft_shift = np.fft.fftshift(fft,axes=[0,1])
	fft_shift *= filt_2D	

	fft = np.fft.ifftshift(fft_shift, axes=[0,1])

	reconstructed=cv.idft(fft,flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
	cv.imshow(reconstructedName,reconstructed)
	cv.imshow(filterName,filt_2D[:,:,0])
	'''
	magnitude_filter = cv.magnitude(fft_shift[:,:,0],fft_shift[:,:,1])
	magnitude_filter = np.log(magnitude_filter+1)
	cv_magnitude = np.array([])
	cv_magnitude = cv.normalize(magnitude_filter,cv_magnitude,0,1,cv.NORM_MINMAX)
	'''

	##cv.getTrackbarPos('Radius',spectrumMagName)
	##cv.getTrackbarPos('Radius',filterName)


	key = cv.waitKey(1) & 0xFF
	if key == ord('e'):
		break