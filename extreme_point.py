import numpy as np
import cv2
import imutils

lower = np.array([0,60,80])
upper = np.array([20,140,250])
frame = cv2.imread('F3.png')

hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower, upper)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

dil = cv2.dilate(mask, kernel, iterations=1)

cv2.imshow('blah1',dil)
im2, contours, hierarchy = cv2.findContours(dil,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key = cv2.contourArea)
xl = tuple(cnt[cnt[:, :, 0].argmin()][0])
xr = tuple(cnt[cnt[:, :, 0].argmax()][0])
yt = tuple(cnt[cnt[:, :, 1].argmin()][0])
yb = tuple(cnt[cnt[:, :, 1].argmax()][0])

print xl,xr,yt,yb
#print xr[0]
#print xl[0]
#print yt[0]
#print yb[0]
y=xr[0]-xl[0]
cropped = dil[yt[1]:2*y,xl[0]:xr[0]]
frame = frame[yt[1]:2*y,xl[0]:xr[0]]
cv2.imshow('blah',frame)

#print 'old',xl,xr,yt,yb

im2, contours, hierarchy = cv2.findContours(cropped,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key = cv2.contourArea)
M = cv2.moments(cnt)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
c=(cx,cy)
print cx
print cy
print c

xl = tuple(cnt[cnt[:, :, 0].argmin()][0])
xr = tuple(cnt[cnt[:, :, 0].argmax()][0])
yt = tuple(cnt[cnt[:, :, 1].argmin()][0])
yb = tuple(cnt[cnt[:, :, 1].argmax()][0])
cv2.circle(frame,c,5,[0,0,255],-1)#centre red circle
print xl,xr,yt,yb
#print xr[0],xl[0],yb[1]
print 'width_L =',xr[0]-cx
print 'width_R =',cx-xl[0]
print 'height=',yb[1]-cy

cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 3)
cv2.circle(frame, xl, 8, (0, 0, 255), -1)#red circle
cv2.circle(frame, xr, 8, (0, 255, 0), -1)#green circle
cv2.circle(frame, yt, 8, (255, 0, 0), -1)#blue circle
cv2.circle(frame, yb, 8, (255, 255, 0), -1)#light blue


#out = frame.copy()

cv2.imshow('output',frame)
cv2.imwrite('outputx.png',frame)

cv2.waitKey(0)
cv2.destroyAllWindows()


