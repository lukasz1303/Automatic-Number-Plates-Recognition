
import cv2
import imutils
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir("photos") if isfile(join("photos", f))]
output = []

for i in onlyfiles:
    output.append(str('output/' + i))
j = 0


def find_plates(filename):

    global j

    image = filename
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    img2 = cv2.imread(image, cv2.IMREAD_COLOR)
    size = img.shape

    s = 720/size[0]
    img = cv2.resize(img, (int(size[1]*s), int(size[0]*s)))
    img2 = cv2.resize(img2, (int(size[1]*s), int(size[0]*s)))
    img = cv2.addWeighted(img, 0.8, img, 0, -50)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 20, 40, 40)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 4))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)

    squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
    light = cv2.threshold(light, 0, 255,
                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
                      dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
    gradX = gradX.astype("uint8")

    gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
    thresh = cv2.threshold(gradX, 0, 255,
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

    lpCnt = None
    roi = None
    # loop over the license plate candidate contours
    k = 0
    for c in cnts:
        # compute the bounding box of the contour and then use
        # the bounding box to derive the aspect ratio
        area = cv2.contourArea(c)
        #print(area)
        if area > 100000 or area < 300:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        area2 = cv2.contourArea(approx)
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        #print (ar)
        cv2.drawContours(img2, [approx], -1, (0, 0, 255), 1)
        if len(approx) >= 4:
            if (ar >= 2.3 and ar <= 6) or (ar >= 0.18 and ar <= 0.35) :

                contours_poly = cv2.approxPolyDP(c, 3, True)
                boundRect = cv2.boundingRect(contours_poly)
                drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype=np.uint8)

                #print(area2, boundRect[2]*boundRect[3])
                #print(area2, boundRect[2]*boundRect[3]/2.5)
                if area2 < (boundRect[2]*boundRect[3]/2.5):
                    continue
                if boundRect[2]>2*boundRect[3]:
                    rect = cv2.rectangle(img2, (int(boundRect[0]), int(boundRect[1])),
                                         (int(boundRect[0] + boundRect[2]), int(boundRect[1] + boundRect[3])), (255, 0, 0),
                                         3)
                    lpCnt = approx
                    break

    if (lpCnt is None):
        lpCnt = cnts[0]

    screenCnt=lpCnt


    mask = np.zeros(gray2.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
    new_image = cv2.bitwise_and(img2, img2, mask=mask)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray2[topx:bottomx + 1, topy:bottomy + 1]

    text = pytesseract.image_to_string(Cropped, config='--psm 11')
    print("Detected license plate Number is:", text)
    #img = cv2.resize(img, (1200, 800))
    Cropped = cv2.resize(Cropped, (400, 200))
    #cv2.imshow('car', img2)
    #cv2.imshow('Cropped', Cropped)
    cv2.imwrite(output[j], img2)
    j+=1
    #cv2.waitKey(0)
    cv2.destroyAllWindows()


for i in onlyfiles:
    print(str('photos/' + i))
    find_plates(str('photos/' + i))
#find_plates(str('photos/' + 'auto21.jpg'))
cv2.waitKey(0)
cv2.destroyAllWindows()