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
    img2 = cv2.imread(image, cv2.IMREAD_COLOR)
    size = img.shape

    s = 720 / size[0]
    img = cv2.resize(img, (int(size[1] * s), int(size[0] * s)))
    img2 = cv2.resize(img2, (int(size[1] * s), int(size[0] * s)))
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
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:100]

    screen_cnt = None
    for c in cnts:

        area = cv2.contourArea(c)
        # print(area)
        if area > 100000 or area < 300:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        area2 = cv2.contourArea(approx)
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        # print (ar)
        cv2.drawContours(img2, [approx], -1, (0, 0, 255), 1)
        if len(approx) >= 4:
            if (2.3 <= ar <= 6) or (0.18 <= ar <= 0.35):

                contours_poly = cv2.approxPolyDP(c, 3, True)
                bound_rect = cv2.boundingRect(contours_poly)

                # print(area2, bound_rect[2]*bound_rect[3])
                # print(area2, bound_rect[2]*bound_rect[3]/2.5)
                if area2 < (bound_rect[2] * bound_rect[3] / 2.5):
                    continue
                if bound_rect[2] > 2 * bound_rect[3]:
                    cv2.rectangle(img2, (int(bound_rect[0]), int(bound_rect[1])), (int(bound_rect[0] + bound_rect[2]),
                                                                                   int(bound_rect[1] + bound_rect[3])),
                                  (255, 0, 0), 3)
                    screen_cnt = approx
                    break

    if screen_cnt is None:
        screen_cnt = cnts[0]

    mask = np.zeros(gray2.shape, np.uint8)
    _ = cv2.drawContours(mask, [screen_cnt], 0, 255, -1, )
    _ = cv2.bitwise_and(img2, img2, mask=mask)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    cropped = gray2[topx:bottomx + 1, topy:bottomy + 1]
    size = cropped.shape
    s = 360 / size[0]
    cropped = cv2.resize(cropped, (int(size[1] * s), int(size[0] * s)))
    cropped = cv2.bilateralFilter(cropped, 15, 25, 25)
    cropped = cv2.addWeighted(cropped, 1.2, cropped, 0, -10)
    cropped = cv2.bilateralFilter(cropped, 15, 25, 25)

    (_, blackAndWhiteImage) = cv2.threshold(cropped, 120, 255, cv2.THRESH_BINARY)

    # cropped = rotate_image(cropped, 1)

    text = pytesseract.image_to_string(blackAndWhiteImage, config='--psm 11')
    print("Detected license plate Number is:", text)

    if zapis:
        cv2.imwrite(output, img2)
    else:
        cv2.imshow('car', img2)
        cv2.imshow('cropped', blackAndWhiteImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def save():
    n = 0
    for i in onlyfiles:
        print(str('photos/' + i))
        find_plates(str('photos/' + i), output[n], True)
        n += 1


def show(nr):
    if nr < 10:
        nr = '0' + str(nr)
    print(str('photos/' + 'auto' + nr + '.jpg'))
    find_plates(str('photos/' + 'auto' + nr + '.jpg'))


# show(4)
save()
