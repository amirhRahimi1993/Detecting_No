import os
import cv2
import numpy as np
import pickle
import multiprocessing as mp
from glob import glob
import random
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import shutil
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import pywt
def find_extermum():
    pass
def create_feture_vector(centers,labels,MAX_SIZE):
    feature_vector = []
    max_lbls = []
    for i in range(len(np.unique(labels))):
        max_lbls.append([len(np.argwhere(labels==i)),i])
        max_lbls.sort(reverse=True)
    for i in range(len(max_lbls)):
        All_pixels = max_lbls[i][0]/MAX_SIZE
        normalized_blue= centers[max_lbls[i][1]][0]/255.0
        normalized_blue_red = centers[max_lbls[i][1]][0] / centers[max_lbls[i][1]][1]
        normalized_blue_green = centers[max_lbls[i][1]][0] / centers[max_lbls[i][1]][2]
        feature_vector.extend([All_pixels*normalized_blue,All_pixels*normalized_blue_red,All_pixels*normalized_blue_green])
    return feature_vector
def wavelet_transform(img):
    coeffs2 = pywt.dwt2(img, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    return [np.mean(LH),np.mean(HL),np.mean(HH)]
def X(address):
    img = cv2.imread(address)
    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 5
    _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    wavelet_vector = wavelet_transform(img)
    feature_vector = create_feture_vector(center,label,len(Z))
    feature_vector = feature_vector + wavelet_vector
    return feature_vector
def simple_mlp(all_feature,lbl):
    trainX, testX, trainY, testY = train_test_split(all_feature, lbl, test_size=0.3)
    #sc = StandardScaler()

    #scaler = sc.fit(trainX)
    #trainX_scaled = scaler.transform(trainX)
    #testX_scaled = scaler.transform(testX)
    mlp_clf = MLPClassifier(hidden_layer_sizes=(12, 6, 2),
                       max_iter=300, activation='tanh',
                            solver='adam')
    mlp_clf.fit(trainX, trainY)
    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(mlp_clf, open(filename, 'wb'))

    # some time later...

    # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))
    # y_pred = loaded_model.score(testX, testY)
    y_pred = mlp_clf.predict(testX)
    print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))
    # fig = plot_confusion_matrix(mlp_clf, testX, testY, display_labels=mlp_clf.classes_)
    # fig.figure_.suptitle("Confusion Matrix for Winequality Dataset")
    # plt.show()
def svm(all_feature,lbl):
    trainX, testX, trainY, testY = train_test_split(all_feature, lbl, test_size=0.5)
    # pca = PCA(n_components=15)
    # trainX=pca.fit(trainX)
    # testX = pca.fit(testX)
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto',kernel="rbf"))
    clf.fit(trainX, trainY)
    filename = 'finalized_model_svm.sav'
    pickle.dump(clf, open(filename, 'wb'))
    print(clf.score(testX,testY))
    fig = plot_confusion_matrix(clf, testX, testY, display_labels=clf.classes_)
    fig.figure_.suptitle("Confusion Matrix for Winequality Dataset")
    plt.show()
def deep_learning_preparing(noises,Pelak,noises_name,Pelak_name):
    noise_cluster = 5
    pelak_clusters=2
    kmeans_noise = KMeans(n_clusters=noise_cluster, random_state=0).fit(noises)
    kmeans_pelak = KMeans(n_clusters=pelak_clusters, random_state=0).fit(Pelak)
    deep_dataset = ["deep_dataset/train/{0}".format(i) for i in range(noise_cluster+pelak_clusters)]+["deep_dataset/test/{0}".format(i) for i in range(noise_cluster+pelak_clusters)]
    for d in deep_dataset:
        os.makedirs(d)
    for k_index in range(len(kmeans_noise.labels_)):
        name = noises_name[k_index].split("/")[-1]
        if random.randint(0,100)<30:
            shutil.copyfile(noises_name[k_index],"deep_dataset/test/{0}/{1}".format(kmeans_noise.labels_[k_index],name))
        else:
            shutil.copyfile(noises_name[k_index], "deep_dataset/train/{0}/{1}".format(kmeans_noise.labels_[k_index],name))
    for k_index in range(len(kmeans_pelak.labels_)):
        name = Pelak_name[k_index].split("/")[-1]
        if random.randint(0,100)<30:
            shutil.copyfile(Pelak_name[k_index],"deep_dataset/test/{0}/{1}".format(kmeans_pelak.labels_[k_index]+noise_cluster,name))
        else:
            shutil.copyfile(Pelak_name[k_index], "deep_dataset/train/{0}/{1}".format(kmeans_pelak.labels_[k_index]+noise_cluster,name))
def main():
    paths = ["datasets/noise","datasets/Pelak"]
    repeator = {}
    for FOLD_INDEX in range(8):
        all_feaures = []
        lbl = []
        Pelak = []
        noises = []
        pelak_name = []
        noises_name =[]
        #pool = mp.Pool(mp.cpu_count())

        for p in paths:
            #results = pool.map(X, [g for g in glob(os.path.join(p, "*.jpeg"))])
            for g in glob(os.path.join(p, "*.jpeg")):
                Feature_vector = X(g)
                if "Pelak" in p:
                    pelak_name.append(g)
                    Pelak.append(Feature_vector)
                    all_feaures.append(Feature_vector)
                    lbl.append(1)
                else:
                    all_feaures.append(Feature_vector)
                    lbl.append(0)
                    #noises.append(Feature_vector)
                    #noises_name.append(g)
                    # if g in repeator.keys():
                    #     continue
                    # if random.randint(0,30) < 10 and len(all_feaures) <800:
                    #     repeator[g] = True
                    #     all_feaures.append(Feature_vector)
                    #     lbl.append(0)

        svm(all_feaures,lbl)
        exit()
    #simple_mlp(all_feaures,lbl)
    #deep_learning_preparing(noises,Pelak,noises_name,pelak_name)
if __name__ == '__main__':
    main()