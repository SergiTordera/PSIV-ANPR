
import cv2
import numpy as np
import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split , KFold
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, roc_curve, RocCurveDisplay, auc
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor, SGDClassifier
from sklearn.svm import SVC , LinearSVC # "Support vector classifier"
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from time import time
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV 
from sklearn.metrics import make_scorer, roc_auc_score
from scipy import stats
from sklearn.svm import SVC , LinearSVC
import pickle
from scipy.ndimage import zoom

# Uso del ejemplo

def blanc_out(img):
    ### treiem part blanca

    hh,ww=img.shape[:2]
    recorte=13
    recorte_top=13
    recorte_lado_r=30
    recorte_lado_l=30
    img= img[recorte_top:hh-recorte,recorte_lado_l:ww-recorte_lado_r]
    return img

def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)


        ### treiem part blanca

        hh,ww=out.shape[:2]
        recorte=13
        recorte_top=25
        recorte_lado_r=18
        recorte_lado_l=21
        out= out[recorte_top:hh-recorte,recorte_lado_l:ww-recorte_lado_r]

        ###
    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        # trim_top = ((out.shape[0] - h) // 2)
        # trim_left = ((out.shape[1] - w) // 2)

        #treiem part blanca
        hh,ww=out.shape[:2]
        recorte=13
        recorte_top=25
        recorte_lado_r=18
        recorte_lado_l=21
        out= out[recorte_top:hh-recorte,recorte_lado_l:ww-recorte_lado_r]
        ###
    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out




def calculate_bbps(image, num_blocks):
    # Convertir la imagen a escala de grises si no lo está
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Dividir la imagen en bloques del tamaño adecuado
    h, w = image.shape
    bh = h // num_blocks
    bw = w // num_blocks

    bbps_features = []

    for i in range(num_blocks):
        for j in range(num_blocks):
            block = image[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            bbps_feature = np.sum(block < 128)  # Contar píxeles menores a 128 (osigui que contem negres)
            bbps_features.append(bbps_feature)

    return bbps_features

def entrena_model(X,y,nom_model):
    # Inicializar el clasificador SVM
    svm_classifier = SVC(kernel='linear')  # Puedes cambiar el kernel según tus necesidades

    # Entrenar el clasificador
    svm_classifier.fit(X, y)
    print(svm_classifier.score(X,y))
    # save the model to disk
    filename = nom_model+"_v5(7)"+".sav"
    pickle.dump(svm_classifier, open(filename, 'wb'))


# CODI PER DIGITS

vectores_de_caracteristicas=[]
etiquetas=[]
num_blocks = 7

input_images_path = "Fuente Matricula/digits/"
files_names = os.listdir(input_images_path)
for files_name in files_names:
    etiquetas.append(files_name[0])
    #detect_image(input_images_path+files_name)
    imagen = cv2.imread(input_images_path+files_name)
    zm_img=clipped_zoom(imagen, 2)
    bbps_features = calculate_bbps(zm_img, num_blocks)

    vectores_de_caracteristicas.append(bbps_features)


import os

def listar_subdirectorios(ruta):
    subdirectorios = [nombre for nombre in os.listdir(ruta) if os.path.isdir(os.path.join(ruta, nombre))]
    return subdirectorios

input_images_path = "digits kaggle/Digits/"
subdirectorios = listar_subdirectorios(input_images_path)

for subdirectorio in subdirectorios:
    input_images_path = "digits kaggle/Digits/"+subdirectorio+"/"
    files_names = os.listdir(input_images_path)
    for files_name in files_names:
        etiquetas.append(subdirectorio)
        #detect_image(input_images_path+files_name)
        imagen = cv2.imread(input_images_path+files_name)
        zm_img=blanc_out(imagen) #no aploquem zoom nomes treiem part blanca
        bbps_features = calculate_bbps(zm_img, num_blocks)
        vectores_de_caracteristicas.append(bbps_features)

print(vectores_de_caracteristicas)
print(etiquetas)

X = np.array(vectores_de_caracteristicas)
y = np.array(etiquetas)
    
entrena_model(X,y,"digits")

# CODI PER LLETRES

vectores_de_caracteristicas=[]
etiquetas=[]

input_images_path = "Fuente Matricula/leters/"
files_names = os.listdir(input_images_path)
for files_name in files_names:
    etiquetas.append(files_name[0])
    #detect_image(input_images_path+files_name)
    imagen = cv2.imread(input_images_path+files_name)
    zm_img=clipped_zoom(imagen, 2)
    bbps_features = calculate_bbps(zm_img, num_blocks)
    vectores_de_caracteristicas.append(bbps_features)
print(vectores_de_caracteristicas)
print(etiquetas)

X = np.array(vectores_de_caracteristicas)
y = np.array(etiquetas)
    
entrena_model(X,y,"lletres")

    