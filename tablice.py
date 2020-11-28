import cv2
import imutils
import numpy as np
import pytesseract
from os import listdir
from os.path import isfile, join

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

onlyfiles = [f for f in listdir("photos") if isfile(join("photos", f))]
output = []

for i in onlyfiles:
    output.append(str('output/' + i))
j = 0


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def find_plates(filename, output=None, zapis=False):
    image = filename
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    size = img.shape

    s = 720 / size[0]
    img = cv2.resize(img, (int(size[1] * s), int(size[0] * s)))
    img2 = img.copy()
    img3 = img.copy()
    img = cv2.addWeighted(img, 0.8, img, 0, -50)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 20, 40, 40)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 4))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kern)

    square_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, square_kern)
    light = cv2.threshold(light, 0, 255,
                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_x = np.absolute(grad_x)
    (minVal, maxVal) = (np.min(grad_x), np.max(grad_x))
    grad_x = 255 * ((grad_x - minVal) / (maxVal - minVal))
    grad_x = grad_x.astype("uint8")

    grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
    grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kern)
    thresh = cv2.threshold(grad_x, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=3)
    thresh = cv2.bitwise_and(thresh, thresh, mask=light)
    thresh = cv2.dilate(thresh, None, iterations=5)
    thresh = cv2.erode(thresh, None, iterations=1)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]

    screen_cnt, cord_x, cord_y, last, bound_rect = find_contour_plate(cnts, 0)

    if screen_cnt is None:
        screen_cnt = cnts[0]

    area = cv2.contourArea(screen_cnt)

    mask = np.zeros(gray2.shape, np.uint8)
    _ = cv2.drawContours(mask, [screen_cnt], 0, 255, -1, )

    cropped, cropped2 = crop_image(mask, gray2, img3)

    cropped = cv2.bilateralFilter(cropped, 15, 25, 25)
    cropped = cv2.addWeighted(cropped, 1.2, cropped, 0, -10)
    cropped = cv2.bilateralFilter(cropped, 15, 25, 25)

    limit = np.mean([np.mean(cropped), 120, 120])

    (_, blackAndWhiteImage) = cv2.threshold(cropped, limit, 255, cv2.THRESH_BINARY)
    edged = cv2.Canny(blackAndWhiteImage, 30, 200)
    edged = cv2.dilate(edged, None, iterations=4)

    cnts2 = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)
    cnts2 = sorted(cnts2, key=lambda ctr: cv2.boundingRect(ctr)[0])[:50]

    letters = find_contour_letters(cnts2, cropped, cropped2, limit)

    if len(letters) == 0:
        cnts2 = sorted(cnts2, key=cv2.contourArea, reverse=True)[:20]
        mask = np.zeros(cropped.shape, np.uint8)
        _ = cv2.drawContours(mask, [cnts2[0]], 0, 255, -1, )

        cropped, cropped2 = crop_image(mask, cropped, cropped2, 25)

        cropped = cv2.bilateralFilter(cropped, 15, 25, 25)
        cropped = cv2.addWeighted(cropped, 1.2, cropped, 0, -10)
        cropped = cv2.bilateralFilter(cropped, 15, 25, 25)

        limit = np.mean([np.mean(cropped), 120, 120])

        (_, blackAndWhiteImage) = cv2.threshold(cropped, limit, 255, cv2.THRESH_BINARY)
        edged = cv2.Canny(blackAndWhiteImage, 30, 200)
        edged = cv2.dilate(edged, None, iterations=4)

        cnts2 = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts2 = imutils.grab_contours(cnts2)
        cnts2 = sorted(cnts2, key=lambda ctr: cv2.boundingRect(ctr)[0])

        letters = find_contour_letters(cnts2, cropped, cropped2, limit)

    text = recognize_letters(letters, zapis)

    if text != '' and not(len(text) == 1 and text not in '0123456789ABCDEFGHJKLMNOPRSTUVWXYZ') \
            and screen_cnt is not None:
        write_on_img(text, img2, cord_x, cord_y)
        cv2.rectangle(img2, (int(bound_rect[0]), int(bound_rect[1])), (int(bound_rect[0] + bound_rect[2]),
                                                                       int(bound_rect[1] + bound_rect[3])),
                      (255, 0, 0), 3)
        print("Detected license plate Number is:", text)

    last_last = 0
    while last < 20:
        c1, c2 = cropped, cropped2
        last, cropped, cropped2 = find_next(cnts, last, area, gray2, img2, img3, limit, zapis)
        if cropped is None:
            cropped, cropped2 = c1, c2
        if last == last_last:
            break
        last_last = last

    if zapis:
        cv2.imwrite(output, img2)
    else:
        cv2.imshow('car', img2)
        cv2.imshow('cropped', cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def recognize_letters(letters, zapis):
    text = ''
    for (l, nr) in zip(letters, range(len(letters))):
        let = pytesseract.image_to_string(l, config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPRSTUVWXYZ "
                                                    "--psm 8")[0]
        if let not in '0123456789ABCDEFGHIJKLMNOPRSTUVWXYZ':
            let = pytesseract.image_to_string(l, config="-c tessedit_char_whitelist"
                                                        "=0123456789ABCDEFGHIJKLMNOPRSTUVWXYZ --psm 6")[0]
        if let not in '0123456789ABCDEFGHIJKLMNOPRSTUVWXYZ':
            let = 'I'
        text += let
        if not zapis:
            cv2.imshow(str(nr), l)

    return text


def find_contour_letters(cnts2, cropped, cropped2, limit):
    letters = []

    for c in cnts2:

        cv2.drawContours(cropped2, cnts2, -1, (0, 0, 255), 2)
        bound_rect = cv2.boundingRect(c)
        if bound_rect[0] < 1 or bound_rect[1] < 1 or bound_rect[0] / bound_rect[1] > 1000 or bound_rect[1] / \
                bound_rect[0] > 1000:
            continue
        if bound_rect[2] * 1.2 < bound_rect[3] < 8 * bound_rect[2]:
            if bound_rect[2] * bound_rect[3] > 2000:
                cv2.rectangle(cropped2, (int(bound_rect[0]), int(bound_rect[1])), (int(bound_rect[0] + bound_rect[2]),
                                                                                   int(bound_rect[1] + bound_rect[3])),
                              (255, 0, 0), 3)
                mask2 = np.zeros(cropped.shape, np.uint8)
                _ = cv2.drawContours(mask2, [c], 0, 255, -1, )

                letter, letter_org = crop_image2(mask2, cropped, cropped2)
                if letter is None:
                    continue

                (_, blackAndWhiteImage) = cv2.threshold(letter, limit, 255, cv2.THRESH_BINARY)
                letters.append(blackAndWhiteImage)
    return letters


def find_contour_plate(cnts, last):
    screen_cnt, bound_rect2 = None, None
    cord_x, cord_y = 0, 0

    for c in cnts:
        last += 1
        area = cv2.contourArea(c)
        if area > 100000 or area < 1300:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        area2 = cv2.contourArea(approx)
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)

        if len(approx) >= 4:
            if 2.3 <= ar <= 6:
                contours_poly = cv2.approxPolyDP(c, 3, True)
                bound_rect = cv2.boundingRect(contours_poly)
                if area2 < (bound_rect[2] * bound_rect[3] / 2.5):
                    continue
                if bound_rect[2] > 1.5 * bound_rect[3]:
                    bound_rect2 = cv2.boundingRect(contours_poly)
                    screen_cnt = approx
                    cord_x, cord_y = bound_rect[0], bound_rect[1]
                    break

    return screen_cnt, cord_x, cord_y, last, bound_rect2


def crop_image(mask, gray2, img3, margin=0):
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x)+margin, np.min(y)+margin)
    (bottomx, bottomy) = (np.max(x)-margin, np.max(y)-margin)
    cropped = gray2[topx:bottomx + 1, topy:bottomy + 1]
    cropped2 = img3[topx:bottomx + 1, topy:bottomy + 1]

    size = cropped.shape
    if size[0] == 0 or size[1] == 0:
        cropped, cropped2 = crop_image(mask, gray2, img3)
    size = cropped.shape
    s = 360 / size[0]

    cropped2 = cv2.resize(cropped2, (int(size[1] * s), int(size[0] * s)))
    cropped = cv2.resize(cropped, (int(size[1] * s), int(size[0] * s)))
    return cropped, cropped2


def crop_image2(mask, gray2, img3):
    (x, y) = np.where(mask == 255)
    topx = np.min(x) - 10 if np.min(x) - 10 > 0 else 1
    topy = np.min(y) - 10 if np.min(y) - 10 > 0 else 1

    (bottomx, bottomy) = (np.max(x) + 10, np.max(y) + 10)
    cropped = gray2[topx:bottomx + 1, topy:bottomy + 1]
    cropped2 = img3[topx:bottomx + 1, topy:bottomy + 1]

    size = cropped.shape
    area = size[0]*size[1]
    if size[0] == 0:
        size = (size[1] * 1.4, size[1])
    if size[1] == 0:
        size = (size[0], size[0] / 1.4)
    if area < 9000 or area < 13000 and size[0] < 3.5*size[1] or size[0] < 1.6*size[1] and area < 16000:
        return None, None
    s = 360 / size[0]
    cropped2 = cv2.resize(cropped2, (int(size[1] * s), int(size[0] * s)))
    cropped = cv2.resize(cropped, (int(size[1] * s), int(size[0] * s)))

    return cropped, cropped2


def write_on_img(text, img, cord_x, cord_y):
    cv2.putText(img, text, (cord_x, cord_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(img, text, (cord_x, cord_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


def save():
    n = 0
    for i in onlyfiles:
        print(str('photos/' + i))
        find_plates(str('photos/' + i), output[n], True)
        n += 1


def show(nr):
    if nr < 10:
        nr = '0' + str(nr)
    print(str('photos/' + 'auto' + str(nr) + '.jpg'))
    find_plates(str('photos/' + 'auto' + str(nr) + '.jpg'))


def preprocessing_plate(mask, cropped, cropped2, margin=0):
    cropped, cropped2 = crop_image(mask, cropped, cropped2, margin)

    cropped = cv2.bilateralFilter(cropped, 15, 25, 25)
    cropped = cv2.addWeighted(cropped, 1.2, cropped, 0, -10)
    cropped = cv2.bilateralFilter(cropped, 15, 25, 25)

    limit = np.mean([np.mean(cropped), 120, 120])

    (_, blackAndWhiteImage) = cv2.threshold(cropped, limit, 255, cv2.THRESH_BINARY)
    edged = cv2.Canny(blackAndWhiteImage, 30, 200)
    edged = cv2.dilate(edged, None, iterations=4)

    cnts2 = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)
    cnts2 = sorted(cnts2, key=lambda ctr: cv2.boundingRect(ctr)[0])

    return cnts2, cropped, cropped2


def find_next(cnts, last, area, gray2, img2, img3, limit, zapis):
    screen_cnt, cord_x, cord_y, last, bound_rect = find_contour_plate(cnts[last:], last)

    if screen_cnt is None:
        if len(cnts) == 0:
            return 0
        screen_cnt = cnts[0]
    cropped, cropped2 = None, None
    if area < 1.6 * cv2.contourArea(screen_cnt) and bound_rect is not None:

        mask = np.zeros(gray2.shape, np.uint8)
        _ = cv2.drawContours(mask, [screen_cnt], 0, 255, -1, )

        cnts2, cropped, cropped2 = preprocessing_plate(mask, gray2, img3)

        letters = find_contour_letters(cnts2, cropped, cropped2, limit)
        if len(letters) == 0:
            cnts2 = sorted(cnts2, key=cv2.contourArea, reverse=True)[:20]
            mask = np.zeros(cropped.shape, np.uint8)
            _ = cv2.drawContours(mask, [cnts2[0]], 0, 255, -1, )

            cnts2, cropped, cropped2 = preprocessing_plate(mask, cropped, cropped2, 25)
            letters = find_contour_letters(cnts2, cropped, cropped2, limit)

        text = recognize_letters(letters, zapis)

        if text != '' and not(len(text) == 1 and text not in '0123456789ABCDEFGHJKLMNOPRSTUVWXYZ') \
                and screen_cnt is not None:
            cv2.rectangle(img2, (int(bound_rect[0]), int(bound_rect[1])), (int(bound_rect[0] + bound_rect[2]),
                                                                           int(bound_rect[1] + bound_rect[3])),
                          (255, 0, 0), 3)
            if text != '' and screen_cnt is not None:
                write_on_img(text, img2, cord_x, cord_y)
            print("Detected license plate Number is:", text)
    return last, cropped, cropped2


show(23)
# save()
