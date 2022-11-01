import copy

import cv2
import numpy as np
import os
from glob import glob
from sklearn.cluster import KMeans
from collections import deque
import time
from threading import Thread
# Initializing a queue
import multiprocessing as mp
q = deque()
thereshold_lower = 30
thereshold_upper = 500
blue_thereshold_coef = 1.2
def min_length(A,cluster):
    min_dist =  19900000
    min_cluster = 0
    candidate_blue= 3
    selected_cluster = []
    coef_index = []
    for i in range(len(cluster)):
        coef_index.append([cluster[i][0]/max(cluster[i][1],cluster[i][2]),i])
    coef_index.sort(reverse=True)
    for i in range(candidate_blue):
        selected_cluster.append(np.uint8(cluster[coef_index[i][1]]))
    for i in range(len(cluster)):
        dist = 0
        for k in range(3):
            dist+= ((cluster[i][k]-A[k])**2)
        if dist < min_dist:
            min_dist = dist
            min_cluster = cluster[i]
    for k in range(len(min_cluster)):
        min_cluster[k] = np.uint8(min_cluster[k])
    its_value = False
    for x in selected_cluster:
        if np.all(min_cluster == x)==True:
            its_value= True
    return min_cluster if its_value else np.asarray([0, 0, 0])
def coef_builder():
    coefs = []
    for i in range(256):
        coefs.append(1.35)#1 + 1 / np.log10(float(i))
    coefs[0]=10
    coefs[1]=4.8
    return coefs
def rectangle_finder(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 1, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    print("Number of contours detected:", len(contours))

    for cnt in contours:
        x1, y1 = cnt[0][0]
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w) / h
            if ratio >= 0.9 and ratio <= 1.1:
                img = cv2.drawContours(img, [cnt], -1, (0, 255, 255), 3)
                cv2.putText(img, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.putText(img, 'Rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                img = cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)

    cv2.imshow("Shapes", img)
    cv2.waitKey(0)
def eliminate_non_blue_values(index,start,end,result,coefs):
    blue_great = []
    process_number = 0
    END = min(end, len(index))
    for INDEX in range(start,END):
        I= index[INDEX]
        process_number+=1
        i,j = I[0],I[1]
        coef= coefs[(np.max(result[i][j]))]
        if result[i][j][0] < coef * max(result[i][j][1],result[i][j][2]):
            continue
        blue_great.append([i,j])
    print(process_number)
    return blue_great
def color_preprocess(img,threshold_color_A,threshold_color_B,coefs):
    # It converts the BGR color space of image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Threshold of blue in HSV space
    lower_blue = np.array(threshold_color_A)
    upper_blue = np.array(threshold_color_B)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(img, img, mask=mask)
    t = time.time()
    X= np.swapaxes(result,1,2)
    X = np.swapaxes(X, 0, 1)
    X=np.bitwise_and(X[0] > 1.35 * X[1], X[0] > 1.35 * X[2])
    X=X.astype(np.uint8)
    result = cv2.bitwise_and(result, result, mask=X)
    kernel = np.ones((3, 3), np.uint8)
    result = cv2.erode(result, kernel)
    return result
def cluster_color(img,number_of_cluster=10):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    indexs = cv2.findNonZero(gray_image)
    index =indexs.squeeze()
    non_zero_values = img[index[:,1],index[:,0],:]
    X =KMeans(n_clusters=number_of_cluster, random_state=0).fit(non_zero_values)
    for i in range(len(index)):
        img[index[i,1]][index[i,0]] = min_length(img[index[i,1]][index[i,0]][:],X.cluster_centers_)
    return img , index
def dfs_from_place(img,I,J,value,eliminate):
    Queue= deque()
    movement=list(range(-5,6))
    Queue.append([I,J])
    X= []
    visit =set()
    QUEUE_ADDED = set()
    while len(Queue)!=0:
        A = Queue.popleft()
        I , J = A[0],A[1]
        if (I,J) in visit:
            continue
        visit.add((I,J))
        QUEUE_ADDED.add((I,J))
        img[I][J] = [0,0,0] if eliminate else [255,255,255]
        X.append([I,J])
        for mx in movement:
            for my in movement:
                if (-1<I+mx<len(img)) and (-1 < J+my < len(img[0])) and np.all(img[I+mx][J+my] == value) == True and (((I+mx,J+my) in QUEUE_ADDED)==False):
                    Queue.append([I+mx,J+my])
    print(len(X))
    if len(X)<thereshold_lower or len(X)>thereshold_upper:
        for x in X:
            for k in range(3):
                img[x[0]][x[1]][k] = 0
    return img
def dfs(img,index):
    print("start dfs")
    for I in range(len(index)):
        j,i = index[I][0],index[I][1]
        s = 0
        coef = 1
        for k in range(3):
            s+=img[i][j][k]
            coef*=img[i][j][k]
        if s == 0 or s == (255*3) or (s==255 and coef==0) :
            continue
        if img[i][j][0] < 1.2*(max(img[i][j][1],img[i][j][2])):
            eliminate = True
        else:
            eliminate = False
        img =dfs_from_place(img,i,j,copy.deepcopy(img[i][j]),eliminate)
    return img
def color_pruner(colorful_img,binary,indexs):
    for I in range(len(indexs)):
        j,i = indexs[I][0],indexs[I][1]
        if colorful_img[i][j][0] < 1.3*(max(colorful_img[i][j][1],colorful_img[i][j][2])):
           binary[i][j] = 0
    return binary
def pruning(img,conts):
    for i in range(len(conts)):
        value = 0
        if thereshold_lower <= len(conts[i]) <= thereshold_upper:
            value = 127
            #dfs(img,conts[i])
        for j in range(len(conts[i])):
            img[conts[i][j][0][1]][conts[i][j][0][0]] = value
    return img
def check_area(value,constant_coef_hight=5.2,constant_coef_low=0.9,constant_height=175,constant_width=300):
    width_index =2
    height_index=3
    if value[width_index] > constant_width or value[height_index] > constant_height or not(constant_coef_low<value[width_index]/value[height_index]< constant_coef_hight):
        return False
    return True
def imsave(img,value,name):
    Y_begin = value[0]
    Y_end = value[0] + value[2]
    X_begin= value[1]
    X_end = value[1]+value[3]
    new_img= img[X_begin:X_end,Y_begin:Y_end,:]
    cv2.imwrite("datasets/{0}_{1}_{2}_{3}_{4}.jpeg".format(name,X_begin,X_end,Y_begin,Y_end),new_img)
def connected_components(threshold,img,name):
    analysis = cv2.connectedComponentsWithStats(threshold,
                                                4,
                                                cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis

    # Initialize a new image to
    # store all the output components
    output = np.zeros(threshold.shape, dtype="uint8")
    for i in range(1, totalLabels):

        if check_area(values[i]):
            # Labels stores all the IDs of the components on the each pixel
            # It has the same dimension as the threshold
            # So we'll check the component
            # then convert it to 255 value to mark it white
            componentMask = (label_ids == i).astype("uint8") * 255
            imsave(img,values[i],name)
            # Creating the Final output mask
            output = cv2.bitwise_or(output, componentMask)

    return output
def main():
    counter =-3
    coefs= coef_builder()
    for p in glob("sample-images/*.jpg"):
        start = time.time()
        counter+=3
        img = cv2.imread(p)
        name = p.split("/")[-1].replace(".jpg","")
        cv2.imwrite("simple_filter/white{0}.jpeg".format(counter),img)
        result = color_preprocess(img,[5, 2, 8],[180, 255, 255],coefs)
        #result,index = cluster_color(result,5)
        imgray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 2, 255, 0)
        #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #result=dfs(result,index
        #thresh =color_pruner(result, thresh,index)
        cv2.imwrite("simple_filter/white{0}.jpeg".format(counter + 1), result)
        #thresh= pruning(thresh,contours)
        print("{0}".format(time.time()-start))
        kernel = np.ones((3,5), np.uint8)
        thresh = cv2.erode(thresh,kernel)
        kernel = np.ones((10, 20), np.uint8)
        thresh=cv2.dilate(thresh, kernel,iterations=2)
        # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # thresh = pruning(thresh, contours)
        output = connected_components(thresh,img,name)
        cv2.imwrite("simple_filter/white{0}.jpeg".format(counter+2),output)
        print("FINISH {0}".format(counter))
main()