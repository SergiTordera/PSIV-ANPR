
import cv2
import numpy as np
import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

def recortar_imagen(input_path, output_path, num_filas, num_columnas):
    # Leer la imagen con OpenCV
    imagen = cv2.imread(input_path)
    
    # Obtener dimensiones originales
    alto_original, ancho_original = imagen.shape[:2]
    
    # Calcular el tamaño de cada celda
    ancho_celda = ancho_original // num_columnas
    alto_celda = alto_original // num_filas
    
    # Crear una lista para almacenar las celdas recortadas
    num=0
    ordree=["blanc","blanc","blanc","blanc","blanc","blanc","blanc","blanc","blanc","blanc"
            ,"blanc","blanc","blanc","blanc","blanc","blanc","0","1","2","3","4","5","6","7"
            ,"8","9","blanc","blanc","blanc","blanc","blanc","blanc",'A', 'B', 'C', 'D', 'E', 
            'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
            'V', 'W', 'X', 'Y', 'Z',"blanc"]
    for fila in range(num_filas):
        for columna in range(num_columnas):
            # Calcular las coordenadas de recorte
            izquierda = columna * ancho_celda
            arriba = fila * alto_celda
            derecha = izquierda + ancho_celda
            abajo = arriba + alto_celda
            
            # Recortar la imagen
            celda = imagen[arriba:abajo, izquierda:derecha]
                # Get dimensions
            height, width = celda.shape[:2]
            
            # Calculate the number of pixels to eliminate from the top
            top_pixels_to_eliminate = int(height * 20 / 100)
            right_pixels_to_eliminate = int(width * 10 / 100)
            cropped_image = celda[top_pixels_to_eliminate:, :]
            cv2.imshow("fuente", cropped_image)
            cv2.waitKey(0)

            # Guardar la imagen resultante
            if num>57:
                num=58
            cv2.imwrite("Fuente Matricula/"+ordree[num]+".png", cropped_image)
            num+=1


import numpy as np
from scipy.ndimage import zoom


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

def blanc_out(img):
    ### treiem part blanca

    hh,ww=img.shape[:2]
    recorte=13
    recorte_top=13
    recorte_lado_r=30
    recorte_lado_l=30
    img= img[recorte_top:hh-recorte,recorte_lado_l:ww-recorte_lado_r]
    return img


def calculate_bbps(image, num_blocks):
    # Convertir la imagen a escala de grises si no lo está
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Dividir la imagen en bloques del tamaño adecuado
    h, w = image.shape
    bh = h // num_blocks
    bw = w // num_blocks

    bbps_features = []
    bbps_no_zero=[]
    for i in range(num_blocks):
        for j in range(num_blocks):
            block = image[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            bbps_feature = np.sum(block > 128)  # Contar píxeles mayores a 128
            bbps_features.append(bbps_feature)
            bbps_no_zero.append(np.count_nonzero(block))

    return bbps_features,bbps_no_zero


# Calcular BBPS

recortar_imagen("Fuente Matricula/general.png", 'output.jpg', 10, 10) #nomes executar el primer cop, despres comentar

# num_blocks = 6
# input_images_path = "Fuente Matricula/digits/"
# files_names = os.listdir(input_images_path)
# for files_name in files_names:
#     #detect_image(input_images_path+files_name)
#     # provadetct(input_images_path+files_name)
#     img = cv2.imread(input_images_path+files_name)
#     zm=clipped_zoom(img, 2)
#     cv2.imshow("aa",zm)
#     cv2.waitKey(0)
#     bbps_features, bbps_no_zero = calculate_bbps(zm, num_blocks)
#     print("features",bbps_no_zero)
#     print("no_zero",bbps_no_zero)
#     print("image shape",zm.shape)
#     print(len(bbps_features),len(bbps_no_zero))



# import os

# def listar_subdirectorios(ruta):
#     subdirectorios = [nombre for nombre in os.listdir(ruta) if os.path.isdir(os.path.join(ruta, nombre))]
#     return subdirectorios

# input_images_path = "digits kaggle/Digits/"
# subdirectorios = listar_subdirectorios(input_images_path)

# for subdirectorio in subdirectorios:
#     input_images_path = "digits kaggle/Digits/"+subdirectorio+"/"
#     files_names = os.listdir(input_images_path)
#     for files_name in files_names[0:3]:
#         # etiquetas.append(subdirectorio)
#         #detect_image(input_images_path+files_name)
#         imagen = cv2.imread(input_images_path+files_name)
#         zm_img=blanc_out(imagen)
#         # zm_img=imagen
#         cv2.imshow("aa",zm_img)
#         cv2.waitKey(0)
#         bbps_features, bbps_no_zero = calculate_bbps(zm_img, num_blocks)
# #     img = cv2.imread(input_images_path+files_name)
# #     zm=clipped_zoom(img, 2)
        

# #     bbps_features, bbps_no_zero = calculate_bbps(zm, num_blocks)
#         print("features",bbps_no_zero)
#         print("no_zero",bbps_no_zero)
#         print("image shape",zm_img.shape)
#         print(len(bbps_features),len(bbps_no_zero))