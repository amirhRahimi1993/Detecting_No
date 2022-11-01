
from collections import deque
import time
import cv2
import numpy as np
import pickle
from glob import glob
import random
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pywt
import matplotlib.pyplot as plt
class extract_ROI:
    def __init__(self,img,thereshold_lower=30,thereshold_upper=500,blue_thereshold_coef=1.2):
        self.q = deque()
        self.thereshold_lower = thereshold_lower
        self.thereshold_upper = thereshold_upper
        self.blue_thereshold_coef = blue_thereshold_coef
        self.img=img
        self.outputs = {}
    def min_length(self,A,cluster):
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
    def coef_builder(self):
        coefs = []
        for i in range(256):
            coefs.append(1.35)#1 + 1 / np.log10(float(i))
        coefs[0]=10
        coefs[1]=4.8
        return coefs
    def rectangle_finder(self,img):
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

    def eliminate_non_blue_values(self,index,start,end,result,coefs):
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
    def color_preprocess(self,threshold_color_A,threshold_color_B,coefs):
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array(threshold_color_A)
        upper_blue = np.array(threshold_color_B)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        result = cv2.bitwise_and(self.img, self.img, mask=mask)
        X= np.swapaxes(result,1,2)
        X = np.swapaxes(X, 0, 1)
        X=np.bitwise_and(X[0] > 1.35 * X[1], X[0] > 1.35 * X[2])
        X=X.astype(np.uint8)
        result = cv2.bitwise_and(result, result, mask=X)
        kernel = np.ones((3, 3), np.uint8)
        result = cv2.erode(result, kernel)
        return result
    def cluster_color(self,number_of_cluster=10):
        gray_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        indexs = cv2.findNonZero(gray_image)
        index =indexs.squeeze()
        non_zero_values = self.img[index[:,1],index[:,0],:]
        X =KMeans(n_clusters=number_of_cluster, random_state=0).fit(non_zero_values)
        for i in range(len(index)):
            img[index[i,1]][index[i,0]] = self.min_length(img[index[i,1]][index[i,0]][:],X.cluster_centers_)
        return img , index
    def color_pruner(self,colorful_img,binary,indexs):
        for I in range(len(indexs)):
            j,i = indexs[I][0],indexs[I][1]
            if colorful_img[i][j][0] < 1.3*(max(colorful_img[i][j][1],colorful_img[i][j][2])):
               binary[i][j] = 0
        return binary
    def pruning(self,img,conts):
        for i in range(len(conts)):
            value = 0
            if self.thereshold_lower <= len(conts[i]) <= self.thereshold_upper:
                value = 127
                #dfs(img,conts[i])
            for j in range(len(conts[i])):
                img[conts[i][j][0][1]][conts[i][j][0][0]] = value
        return img
    def __check_area(self,value,constant_coef_hight=5.2,constant_coef_low=0.9,constant_height=175,constant_width=300):
        width_index =2
        height_index=3
        if value[width_index] > constant_width or value[height_index] > constant_height or not(constant_coef_low<value[width_index]/value[height_index]< constant_coef_hight):
            return False
        return True
    def imsave(self,value):
        Y_begin = value[0]
        Y_end = value[0] + value[2]
        X_begin= value[1]
        X_end = value[1]+value[3]
        new_img= self.img[X_begin:X_end,Y_begin:Y_end,:]
        self.outputs["{0}_{1}_{2}_{3}".format(X_begin,X_end,Y_begin,Y_end)]=new_img
    def connected_components(self,threshold):
        analysis = cv2.connectedComponentsWithStats(threshold,4,cv2.CV_32S)
        (totalLabels, label_ids, values, centroid) = analysis
        output = np.zeros(threshold.shape, dtype="uint8")
        for i in range(1, totalLabels):
            if self.__check_area(values[i]):
                componentMask = (label_ids == i).astype("uint8") * 255
                self.imsave(values[i])
                output = cv2.bitwise_or(output, componentMask)
        return output
    def main(self):
        coefs= self.coef_builder()

        result = self.color_preprocess([5, 2, 8],[180, 255, 255],coefs)
        #result,index = cluster_color(result,5)
        imgray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 2, 255, 0)
        kernel = np.ones((3,5), np.uint8)
        thresh = cv2.erode(thresh,kernel)
        kernel = np.ones((10, 20), np.uint8)
        thresh=cv2.dilate(thresh, kernel,iterations=2)
        self.connected_components(thresh)
        return self.outputs
class discriminator:
    def __init__(self,img,ROIs,models):
        self.ROIs = ROIs
        self.img = img
        self.models = models
    def __create_feture_vector(self,centers, labels, MAX_SIZE):
        feature_vector = []
        max_lbls = []
        for i in range(len(np.unique(labels))):
            max_lbls.append([len(np.argwhere(labels == i)), i])
            max_lbls.sort(reverse=True)
        for i in range(len(max_lbls)):
            All_pixels = max_lbls[i][0] / MAX_SIZE
            normalized_blue = centers[max_lbls[i][1]][0] / 255.0
            normalized_blue_red = centers[max_lbls[i][1]][0] / centers[max_lbls[i][1]][1]
            normalized_blue_green = centers[max_lbls[i][1]][0] / centers[max_lbls[i][1]][2]
            feature_vector.extend([All_pixels * normalized_blue, All_pixels * normalized_blue_red, All_pixels * normalized_blue_green])
        return feature_vector

    def wavelet_transform(self):
        coeffs2 = pywt.dwt2(self.img, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2
        return [np.mean(LH), np.mean(HL), np.mean(HH)]
    def __X(self):
        feature_vector=[]
        for k in self.ROIs:
            img = self.ROIs[k]
            Z = img.reshape((-1, 3))
            # convert to np.float32
            Z = np.float32(Z)
            # define criteria, number of clusters(K) and apply kmeans()
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 5
            _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            FEATURE = self.__create_feture_vector(center, label, len(Z)) + self.wavelet_transform()
            feature_vector.append(FEATURE)
        return feature_vector

    def __simple_svm(self,all_feature):
        y_pred = self.models.predict(all_feature)
        return y_pred
    def pullThetrigger(self):
        feature_vector = self.__X()
        if len(feature_vector)==0:
            return []
        prediction = self.__simple_svm(feature_vector)
        counter = 0
        Answers =[]
        for k in self.ROIs.keys():
            if prediction[counter]==1:
                Z=k.split("_")
                for i in range(len(Z)):
                    Z[i]=int(Z[i])
                Answers.append(Z)
            counter+=1
        return Answers
def basir_task_main(img):
    t = time.time()
    filename = 'finalized_model_svm.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    start = time.time()
    e = extract_ROI(img)
    outputs = e.main()
    d = discriminator(img, outputs, loaded_model)
    predicts = d.pullThetrigger()
    name = p.split("/")[-1].replace(".jpg", "")
    for A in predicts:
        color = (50, 50, 255)

        # Line thickness of 2 px
        thickness = 2
        start_point = (A[2], A[0])
        end_point = (A[3], A[1])
        img = cv2.rectangle(img, start_point, end_point, color, thickness)
    print("Calculate time is : {0}".format(time.time() - start))
    return img
# for p in glob("sample-images/*.jpg"):
#     img = cv2.imread(p)
#     name = p.split("/")[-1].replace(".jpg", "")
#
#     cv2.imwrite("result/{0}.jpeg".format(name),basir_task_main(img))
