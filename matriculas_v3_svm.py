import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
import numpy as np
import easyocr
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, roc_curve, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

H = 1000
W = 1600
ASPECT_RATIO = 340/110
TOP_ASPECT_RATIO = ASPECT_RATIO*1.5
LOW_ASPECT_RATIO = ASPECT_RATIO*0.5
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

def dilate(img, kernel_size, iters, show=False):
    k = np.ones(kernel_size)
    img = cv2.dilate(img, k, iters)
    if show:
        cv2.imshow('shapes', img)
        cv2.waitKey(0)
    return img


def erode(img, kernel_size, iters, show=False):
    k = np.ones(kernel_size)
    img = cv2.erode(img, k, iters)
    if show:
        cv2.imshow('shapes', img)
        cv2.waitKey(0)
    return img


def find_region(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    best_contour_area = 0
    best_conour = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, .01 * cv2.arcLength(contour, True), True)
        if len(approx) >= 4:
            x, y, w, h = cv2.boundingRect(contour)
            if LOW_ASPECT_RATIO < w/h < TOP_ASPECT_RATIO:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if w * h > best_contour_area and y > H/4:
                    best_conour = ((x, y), (x + w, y + h))
                    best_contour_area = w*h
    return best_conour


def crop_image(img, best_contour):
    img = cv2.resize(img, [W, H])
    OriginalImage = img.copy()
    cropped_img = OriginalImage[best_contour[0][1]-25:best_contour[1][1]+20, best_contour[0][0]-25:best_contour[1][0]+20]
    cv2.imshow('Croped', cropped_img)
    cv2.waitKey(0)
    return cropped_img

def find_letters(cropped_img, show=True):
    cropped_img_ = cropped_img.copy()
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (75,75))
    smooth = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    division = cv2.divide(gray, smooth, scale=255)
    result = cv2.threshold(division, 0, 255, cv2.THRESH_OTSU )[1]
    image = cv2.bitwise_not(result)
    image = image.astype(np.uint8)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ordered_y = []
    vertices = []
    H1, W1 = image.shape #h,w,c
    for i, contour in enumerate(contours):
        approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
        if len(approx) >=4:
            x, y, w, h = cv2.boundingRect(contour)
            if w < h and 10000 > w*h > 600 and not(x < W1*0.02 or x+w > W1*0.98) and not(y < H1*0.03 or y+h > H1*0.98): #and letter_ratio*0.8 < h/w < letter_ratio*1.2 
                print(f'{w} {h} {w*h} {w/h}')
                cv2.rectangle(cropped_img_, (x, y),(x + w, y + h), (0, 255, 0), 2)
                cv2.putText(cropped_img_, f"{w*h}", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                ordered_y.append((x, y, w, h))
                vertices.append((x, y, x+w, y+h))
    filtered_y = []
    for contour in ordered_y:
        insiide = False
        for v in vertices:
            x = contour[0]
            x2 = x+contour[2]
            if v[2] > x > v[0] and v[2] > x2 > v[0]:
                insiide = True
                break
        if not insiide: filtered_y.append(contour)

    filtered_y.sort(key= lambda x: (x[3], -x[2]), reverse=True)
    filtered_y = filtered_y[1:] if len(filtered_y) > 7 else filtered_y

    letters_images = []
    data = dict()
    for charac in filtered_y[-7:]:
        cv2.rectangle(cropped_img_, (charac[0], charac[1]),(charac[0] + charac[2], charac[1] + charac[3]), (0, 0, 255), 2)
        print(charac)
        #letter = cropped_img[charac[0]:(charac[0]+charac[2]), charac[1]:(charac[1]+charac[3])]
        letter = cropped_img[charac[1]:(charac[1]+charac[3]), charac[0]:charac[0]+charac[2]]
        letters_images.append(letter)
        data[charac[0]] = cropped_img[charac[1]:(charac[1]+charac[3]), charac[0]:charac[0]+charac[2]]
        if show:
            cv2.imshow('letter', letter)
            cv2.waitKey(0)
    if show:
        cv2.imshow('new_img', cropped_img_)
        cv2.waitKey(0)

    x_pos = list(data.keys())
    x_pos.sort()
    ordered_images = [data[x] for x in x_pos]
    return ordered_images

def scan_letter(reader, letter):
    detected_character = None
    img_gray = cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY)
    bin_img = (img_gray<100)*255
    bin_img = bin_img.astype(np.uint8)
    cv2.imshow('letra', bin_img)
    cv2.waitKey(0)
    results = reader.readtext(img_gray)
    if results:
        detected_character = results[0][1]
        print("Detected character:", detected_character)    
    else:
        print("No character detected.")
    return detected_character

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
def acurracy(y_pred_digits,y_pred_letters,cont_acc_l,cont_acc_d,l_acc,d_acc):
    acc_leter_tar=['D','K','P','G','M','Z','G','L','K','J','M','B','J','H','B','L','K','K','C','R','T','G','V','T','D','C','X','J','B','Z','K','J','P','J','R','F',
            'B','N','X','J','B','X','J','T','C','H','G','Y','F','H','C']

    acc_digit_tar=['5','7','9','5',
                '1','5','5','6',
                '0','1','8','2',
                '3','0','4','4',
                '5','7','8','9',
                '6','9','2','9',
                '3','6','6','0',
                '6','0','0','0',
                '3','5','8','7',
                '1','4','9','8',
                '2','3','4','4',
                '0','9','0','7',
                '6','5','5','4',
                '6','4','0','1',
                '8','7','2','7',
                '5','2','7','5',
                '4','6','7','4',]

    for idx,x in enumerate(y_pred_letters):
        # print("avans dentrar if ",cont_acc_l)
        # print(len(acc_leter_tar))
        if x ==acc_leter_tar[cont_acc_l+idx]:
            l_acc=l_acc+1
            # print(l_acc)
    cont_acc_l=cont_acc_l+3
    for idx,x in enumerate(y_pred_digits):
        # print("avans dentrar if ",cont_acc_d)
        # print(len(acc_digit_tar),cont_acc_d+idx)
        # print("gt",acc_digit_tar[cont_acc_d+idx])
        if x ==acc_digit_tar[cont_acc_d+idx]:
            d_acc=d_acc+1

    cont_acc_d=cont_acc_d+4
    return cont_acc_l,cont_acc_d,l_acc,d_acc,acc_leter_tar,acc_digit_tar
    
def predic_letters(letters,show=False):
    carecteristiques=[]
    n_blocks=7
    for l in letters:
        if show:
            cv2.imshow("letter",l)
            cv2.waitKey(0)
        bbps_features = calculate_bbps(l, n_blocks)
        print(bbps_features)
        carecteristiques.append(bbps_features)
    
    
    X_valid_digits = np.array(carecteristiques[0:4])

    X_valid_leters = np.array(carecteristiques[4:])

    loaded_model = pickle.load(open("digits_v4(7).sav", 'rb'))
    y_pred_digits = loaded_model.predict(X_valid_digits)
    # print("prediccions numeros",y_pred_digits)
    # print("'"+y_pred[0]+"','"+y_pred[1]+"','"+y_pred[2]+"',"+y_pred[3]+"',")
    loaded_model = pickle.load(open("lletres_v4(7).sav", 'rb'))
    y_pred_letters = loaded_model.predict(X_valid_leters)
    # print("prediccions lletres",y_pred_letters)
    # print("'"+y_pred[0]+"','"+y_pred[1]+"','"+y_pred[2]+"',")

    print("Matricula: "+y_pred_digits[0]+y_pred_digits[1]+y_pred_digits[2]+y_pred_digits[3]+" "+y_pred_letters[0]+y_pred_letters[1]+y_pred_letters[2])
    return y_pred_digits,y_pred_letters

def detect_mat(img_path):
    main_img = cv2.imread(img_path)
    main_img = cv2.resize(main_img, [W, H])
    gray = cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY)
    
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
    # cv2.imshow("shapes", blackhat)

    blackhat = erode(blackhat, [2, 5], 20)
    blackhat = dilate(blackhat, [5, 20], 50)
    blackhat = erode(blackhat, [20, 5], 50)
    blackhat = erode(blackhat, [10, 5], 20)

    blackhat = dilate(blackhat, [10, 5], 60)
    blackhat = dilate(blackhat, [10, 5], 60)
    blackhat = dilate(blackhat, [5, 10], 60)
    blackhat = dilate(blackhat, [10, 30], 30)

    blackhat = dilate(blackhat, [2, 10], 30)
    blackhat = dilate(blackhat, [2, 10], 25)
    blackhat = dilate(blackhat, [2, 10], 15)

    # Convert 2 Grayscale

    image = (blackhat > 100) * 255
    image = image.astype(np.uint8)
    # cv2.imshow('shapes', image)
    # cv2.waitKey(0)

    main_image_ = main_img.copy()

    # Find Contours
    best_contour = find_region(image)
    cv2.rectangle(main_image_, best_contour[0], best_contour[1], (255, 0, 0), 2)
    # cv2.imshow('mat', main_image_)
    cv2.waitKey(0)

    # Crop Image
    image = crop_image(main_img, best_contour)
    
    # Find Poligons
    
    letters = find_letters(image)

    return letters
    

    # reader = easyocr.Reader(lang_list=['en'], gpu=False)

    # convert_num2char = {"6": "G", "2":"Z"}
    # convert_char2num = {"L":"4"}

    # plate_number = []
    # for l in letters:
    #     char = scan_letter(reader=reader, letter=l)
    #     char = char if char else '-'
    #     if len(plate_number)<4 and char.isalpha():
    #         char = convert_char2num[char]
    #     elif len(plate_number)>=4 and char.isdigit():
    #         char = convert_num2char[char]
    #     plate_number.append(char)
    
    # print("\n\n\n")
    # print("MATRICULA: ", plate_number)



def main(path):

    acc_leter_tar=['D','K','P','G','M','Z','G','L','K','J','M','B','J','H','B','L','K','K','C','R','T','G','V','T','D','C','X','J','B','Z','K','J','P','J','R','F',
            'B','N','X','J','B','X','J','T','C','H','G','Y','F','H','C']

    acc_digit_tar=['5','7','9','5','1','5','5','6','0','1','8','2','3','0','4','4','5','7','8','9','6','9','2','9','3','6','6','0','6','0','0','0','3','5','8','7',
                '1','4','9','8','2','3','4','4','0','9','0','7','6','5','5','4','6','4','0','1','8','7','2','7','5','2','7','5','4','6','7','4']
    total_y_pred_letter=[]
    total_y_pred_digits=[]
    cont_acc_l=0
    cont_acc_d=0
    l_acc=0
    d_acc=0


    files_names = os.listdir(path)
    for files_name in files_names:
        letters=detect_mat(path+files_name)
        y_pred_digits,y_pred_letters=predic_letters(letters)
        total_y_pred_digits.extend(y_pred_digits)
        total_y_pred_letter.extend(y_pred_letters)
        cont_acc_l,cont_acc_d,l_acc,d_acc,acc_leter_tar,acc_digit_tar=acurracy(y_pred_digits,y_pred_letters,cont_acc_l,cont_acc_d,l_acc,d_acc)
    print("Acurracy Letter: ", (l_acc*100)/len(acc_leter_tar))
    print("Acurracy Digits: ", (d_acc*100)/len(acc_digit_tar))
    
    letras_mayusculas_sin_vocales = [chr(i) for i in range(65, 91) if chr(i) not in ['A', 'E', 'I', 'O', 'U', 'Q', 'Ñ']]#,'S','W','Y','Z'

    print(len(total_y_pred_letter),set(total_y_pred_letter))
    print(len(total_y_pred_digits),set(total_y_pred_digits))
    # nuevo_array = [letra for letra in letras_mayusculas_sin_vocales if letra in total_y_pred_letter]

    cm_l=confusion_matrix(acc_leter_tar,total_y_pred_letter,labels=list(set(acc_leter_tar)))#list(set(total_y_pred_letter))
    cm_d=confusion_matrix(acc_digit_tar,total_y_pred_digits,labels=["0","1","2","3","4","5","6","7","8","9"])


    disp = ConfusionMatrixDisplay(confusion_matrix=cm_l, display_labels=list(set(acc_leter_tar)))#list(set(total_y_pred_letter))
    disp.plot(xticks_rotation="horizontal")
    plt.show()

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_d, display_labels=["0","1","2","3","4","5","6","7","8","9"])
    disp.plot(xticks_rotation="horizontal")
    plt.show()
if __name__ == "__main__":
    path = "BBDD Matricules/"
    main(path)


    