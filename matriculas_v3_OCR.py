import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import easyocr

H = 1000
W = 1600
ASPECT_RATIO = 340/110
TOP_ASPECT_RATIO = ASPECT_RATIO*1.5
LOW_ASPECT_RATIO = ASPECT_RATIO*0.5


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


def crop_image(img, best_contour, show=False):
    img = cv2.resize(img, [W, H])
    OriginalImage = img.copy()
    cropped_img = OriginalImage[best_contour[0][1]-25:best_contour[1][1]+20, best_contour[0][0]-25:best_contour[1][0]+20]
    if show:
        cv2.imshow('Croped', cropped_img)
        cv2.waitKey(0)
    return cropped_img


def find_letters(cropped_img, show=False, debug=False):
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
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        if len(approx) >=4 :
            x, y, w, h = cv2.boundingRect(contour)
            if w < h and 10000 > w*h > 600 and not (x < W1*0.02 or x+w > W1*0.98) and not(y < H1*0.03 or y+h > H1*0.98): #and letter_ratio*0.8 < h/w < letter_ratio*1.2 
                if debug: print(f'{w} {h} {w*h} {w/h}')
                cv2.rectangle(cropped_img_, (x, y), (x + w, y + h), (0, 255, 0), 2)
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

    filtered_y.sort(key=lambda x: (x[3], -x[2]), reverse=True)
    filtered_y = filtered_y[1:] if len(filtered_y) > 7 else filtered_y

    letters_images = []
    data = dict()
    for charac in filtered_y[-7:]:
        cv2.rectangle(cropped_img_, (charac[0], charac[1]),(charac[0] + charac[2], charac[1] + charac[3]), (0, 0, 255), 2)
        if debug: print(charac)
        letter = cropped_img[charac[1]-5:(charac[1]+charac[3]+5), charac[0]-5:charac[0]+charac[2]+5]
        letters_images.append(letter)
        data[charac[0]] = cropped_img[charac[1]-5:(charac[1]+charac[3]+5), charac[0]-5:charac[0]+charac[2]+5]
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


def scan_letter(reader, letter, show=False, debug=False):
    detected_character = None
    img_gray = cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY)
    bin_img = (img_gray < 100) * 255
    bin_img = bin_img.astype(np.uint8)
    if show:
        cv2.imshow('letra', bin_img)
        cv2.waitKey(0)
    results = reader.readtext(bin_img)
    if results:
        detected_character = results[0][1]
        if debug: print("Detected character:", detected_character)    
    else:
        if debug: print("No character detected.")
    return detected_character


def detect_mat(img_path, show=False, debug=False):
    main_img = cv2.imread(img_path)
    main_img = cv2.resize(main_img, [W, H])
    gray = cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY)

    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)

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
    if show:
        cv2.imshow('shapes', image)
        cv2.waitKey(0)

    main_image_ = main_img.copy()

    # Find Contours
    best_contour = find_region(image)
    cv2.rectangle(main_image_, best_contour[0], best_contour[1], (255, 0, 0), 2)
    if show:
        cv2.imshow('mat', main_image_)
        cv2.waitKey(0)

    # Crop Image
    image = crop_image(main_img, best_contour)
    
    # Find Poligons
    letters = find_letters(image)

    reader = easyocr.Reader(lang_list=['en'], gpu=False)

    convert_num2char = {"6": "G", "2": "Z", "1": "D", "0": "D"}
    convert_char2num = {"L": "4"}

    plate_number = []
    for l in letters:
        char = scan_letter(reader=reader, letter=l)
        char = char if char else '-'
        if len(plate_number) < 4 and char.isalpha():
            char = convert_char2num[char]
        elif len(plate_number) >= 4 and char.isdigit():
            char = convert_num2char[char]
        plate_number.append(char)
    if debug:
        print("\n\n\n")
        print("MATRICULA: ", plate_number)
    return "".join(plate_number)


def main(path, debug=False):
    RESULTS = [
        "3208HKR",
        "5796DKP", "1556GMZ", "0182GLK",
        "3044JMB", "5789JHB", "6929LKK",
        "3660CRT", "6000GVT", "3587DCX",
        "1498JBZ", "2344KJP", "0907JRF",
        "6554BNX", "6401JBX", "8727JTC",
        "5275HGY", "4674FHC"
    ]
    scores = []
    files_names = os.listdir(path)
    results_y_num = []
    results_y_pred_num = []
    results_y_let = []
    results_y_pred_let = []
    for i, files_name in enumerate(files_names):
        r = detect_mat(path+files_name)
        if debug:
            print(len(r), r)
            print(len(RESULTS[i]), RESULTS[i])
        results_y_num.extend(RESULTS[i][:-3])
        results_y_let.extend(RESULTS[i][-3:])
        results_y_pred_num.extend(r[:-3])
        results_y_pred_let.extend(r[-3:])

    ac_num = accuracy_score(results_y_num, results_y_pred_num)
    ac_let = accuracy_score(results_y_let, results_y_pred_let)
    
    let_labels = list(set(results_y_let))
    num_labels = list(set(results_y_num))

    let_labels.sort()
    num_labels.sort()

    let_labels.append("-")
    num_labels.append("-")

    let_cm = confusion_matrix(y_true=results_y_let, y_pred=results_y_pred_let, labels=let_labels)
    num_cm = confusion_matrix(y_true=results_y_num, y_pred=results_y_pred_num, labels=num_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=let_cm, display_labels=let_labels)
    disp.plot()
    plt.show()

    disp = ConfusionMatrixDisplay(confusion_matrix=num_cm, display_labels=num_labels)
    disp.plot()
    plt.show()

    print("\n\n\n")
    print("ACCURACY NUMBERS: ", ac_num)
    print("ACCURACY NUMBERS: ", ac_let)


if __name__ == "__main__":
    path = "BBDD-Matricules/"
    main(path)