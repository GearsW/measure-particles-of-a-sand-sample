import numpy as np
import cv2
import matplotlib.pyplot as plt
import cv2 as cv
import math

imga = cv2.imread('paper+ chocola + sesam most real scenario.jpg')
img = cv2.bitwise_not(imga) #"inverses colors because the software couldn't work with the black background."

"""Pre-Process might be obsolite now with a better background

dilated_img = cv2.dilate(img, np.ones((5,5), np.uint8)) 
bg_img = cv2.medianBlur(dilated_img, 7)
diff_img = 255 - cv2.absdiff(img, bg_img)

norm_img = diff_img.copy() # Needed for 3.x compatibility
cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

_, thr_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)
cv2.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

"""

"""Process"""
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# removing static
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 5)

# known background
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# known forground "this might be where my software stuggles it's removing the smallest particles and only keeping the biggest ones around (centers are not found)
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)

# finding the unknown
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# label markers
ret, markers = cv2.connectedComponents(sure_fg)
# adding 1 so the background is not 0 but 1
markers = markers+1
# markering unknown with 0
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

""" calculating the area and the contour """

barr = np.uint8(markers)

#a4 scanner is 219mm*297mm (known size)#
known_size = barr.size / float(219*297) 
print(img.size)
print(known_size)

#area calculation 
maxlabels = np.unique(barr)
print("[INFO] {} unique segments found".format(len(np.unique(barr)) - 1))
ret,thresh = cv.threshold(barr,127,255,0)
im2,contours,hierarchy = cv.findContours(thresh, 1, 2)

#making a list of all the particles and calculating the diameters of a circle of every particle
marker_list = [];
area_list = [];

i = 1
while i <= len(maxlabels):
    cnt = contours[i]
    area =  math.sqrt((known_size / cv.contourArea(cnt)) / math.pi)
    i += 1
    marker_list.append(i)
    area_list.append(area)
    cv2.drawContours(barr, [cnt], 0, (0,0,255), 3)
    #print(area)

#printing the list
np.set_printoptions(suppress=True)
titel_tabel_dg = ['deeltje', 'grootte']
tabel_deeltjes_grootte = np.column_stack((marker_list, area_list))

print(titel_tabel_dg)
print(tabel_deeltjes_grootte)

#moving the list around so it calculates smallest to biggest and groups the particles with the same diameter together
marker_list2 = [];
area_list2 = [];

a = area_list

d = {}
for item in a:
    if item in d:
        d[item] = d.get(item)+1
    else:
        d[item] = 1

for k,v in d.items():
    marker_list2.append(k)    
    area_list2.append(v)

list3 = np.array(marker_list2)
list4 = np.array(area_list2)
idx2   = np.argsort(list3)

list3 = np.array(list3)[idx2]
list4 = np.array(list4)[idx2]
voorkomende_deeltjes = np.column_stack((list3, list4))

print(voorkomende_deeltjes)

"""making some basic files for now"""
#histogram of different particles sizes and how many they have
plt.figure()
plt.hist(list3)
plt.xlabel('diameter (mm)')
plt.ylabel('hoeveelheid')
plt.savefig('Area.png')

#pictures for reference and seeing if it works
cv2.imwrite("a.png", img)
cv2.imwrite("b.png", markers)
cv2.imwrite("c.png", sure_fg)