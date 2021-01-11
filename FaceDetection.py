# -*- coding: utf-8 -*-
"""
Created on Tue May 12 20:29:58 2020

@author: rliu25
"""

# -*- coding: utf-8 -*-
import sys

"""
Created on Mon May  4 21:08:20 2020

@author: rliu25
@author: JingJing Zhuang (jzhuang3)

"""
import random

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

import json
import math
import time

WINDOW_SIZE = 24
HALF_WINDOW = WINDOW_SIZE // 2


def data_load(directory_name):
    image_rgb = []
    image_grey = []
    for filename in os.listdir(directory_name):
        image = cv2.imread(directory_name + "/" + filename)
        image_rgb.append(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_grey.append(img)

    return image_rgb, image_grey


def change_size(data):
    data = np.array(data)
    img = cv2.resize(data, (WINDOW_SIZE, WINDOW_SIZE), interpolation=cv2.INTER_AREA)
    return img


def sample_data(imageset, n):
    data = []
    data.extend([change_size(f) for f in random.sample(imageset, n)])
    return np.array(data)


def to_integral(img, x, y):
    result = np.zeros((x, y), dtype=np.float32)
    result = cv2.normalize(img, dst=result, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    sum_ = np.zeros((x + 1, y + 1), dtype=np.float32)
    imageIntegral = cv2.integral(result, sum_, cv2.CV_32FC1)
    return imageIntegral


def position(WINDOW_SIZE, width, hight):
    pos = []
    for x in range(WINDOW_SIZE - width + 1):
        for y in range(WINDOW_SIZE - hight + 1):
            pos.append([x, y])
    return pos


def possible_shapes(width, height, WINDOW_SIZE):
    shape = []
    for w in range(width, WINDOW_SIZE + 1, width):
        for h in range(height, WINDOW_SIZE + 1, height):
            shape.append([w, h])
    return shape


def feature2h(image, x, y, width, height):
    h_w = width // 2
    position = []
    left_value = image[x + h_w, y + height] - image[x, y + height] - image[x + h_w, y] + image[x, y]
    right_value = image[x + width, y + height] - image[x + h_w, y + height] - image[x + width, y] + image[x + h_w, y]
    position.append(x), position.append(y), position.append(width), position.append(height)
    return position, right_value - left_value  # left_value - right_value


def feature2v(image, x, y, width, height):
    h_h = height // 2
    position = []
    top_value = image[x + width, y + h_h] - image[x, y + h_h] - image[x + width, y] + image[x, y]
    bottom_value = image[x + width, y + height] - image[x, y + height] - image[x + width, y + h_h] + image[x, y + h_h]
    position.append(x), position.append(y), position.append(width), position.append(height)
    return position, bottom_value - top_value


def feature3h(image, x, y, width, height):
    h_w = width // 3
    position = []
    left_value = image[x + h_w, y + height] - image[x, y + height] - image[x + h_w, y] + image[x, y]
    mid_value = image[x + 2 * h_w, y + height] - image[x + h_w, y + height] - image[x + 2 * h_w, y] + image[x + h_w, y]
    right_value = image[x + width, y + height] - image[x + 2 * h_w, y + height] - image[x + width, y] + image[
        x + 2 * h_w, y]
    position.append(x), position.append(y), position.append(width), position.append(height)
    return position, mid_value - right_value - left_value


def feature3v(image, x, y, width, height):
    h_h = height // 3
    position = []
    top_value = image[x + width, y + h_h] - image[x, y + h_h] - image[x + width, y] + image[x, y]
    mid_value = image[x + width, y + 2 * h_h] - image[x, y + 2 * h_h] - image[x + width, y + h_h] + image[x, y + h_h]
    bottom_value = image[x + width, y + height] - image[x, y + height] - image[x + width, y + 2 * h_h] + image[
        x, y + 2 * h_h]
    position.append(x), position.append(y), position.append(width), position.append(height)
    return position, mid_value - bottom_value - top_value


def feature4(image, x, y, width, height):
    h_w = width // 2
    h_h = height // 2
    position = []
    top_left = image[x + h_w, y + h_h] - image[x, y + h_h] - image[x + h_w, y] + image[x, y]
    top_right = image[x + width, y + h_h] - image[x + h_w, y + h_h] - image[x + width, y] + image[x + h_w, y]
    bottom_left = image[x + h_w, y + height] - image[x, y + height] - image[x + h_w, y + h_h] + image[x, y + h_h]
    bottom_right = image[x + width, y + height] - image[x + h_w, y + height] - image[x + width, y + h_h] + image[
        x + h_w, y + h_h]
    position.append(x), position.append(y), position.append(width), position.append(height)
    return position, bottom_right + top_left - top_right - bottom_left


def getfea2h(img):
    fea2h = []
    value = []
    for i, (w, h) in enumerate(possible_shapes(1, 2, WINDOW_SIZE)):
        for j, (x, y) in enumerate(position(WINDOW_SIZE, w, h)):
            f = {}
            f_2h, v = feature2h(img, x, y, w, h)
            f['name'] = 'feature2h'
            f['position'] = f_2h
            fea2h.append(f)
            value.append(v)
    return fea2h, value


def getfea2v(img):
    fea2v = []
    value = []
    for i, (w, h) in enumerate(possible_shapes(2, 1, WINDOW_SIZE)):
        for j, (x, y) in enumerate(position(WINDOW_SIZE, w, h)):
            f = {}
            f_2v, v = feature2v(img, x, y, w, h)
            f['name'] = 'feature2v'
            f['position'] = f_2v
            fea2v.append(f)
            value.append(v)
    return fea2v, value


def getfea3h(img):
    fea3h = []
    value = []
    for i, (w, h) in enumerate(possible_shapes(1, 3, WINDOW_SIZE)):
        for j, (x, y) in enumerate(position(WINDOW_SIZE, w, h)):
            f = {}
            f_3h, v = feature3h(img, x, y, w, h)
            f['name'] = 'feature3h'
            f['position'] = f_3h
            fea3h.append(f)
            value.append(v)
    return fea3h, value


def getfea3v(img):
    fea3v = []
    value = []
    for i, (w, h) in enumerate(possible_shapes(3, 1, WINDOW_SIZE)):
        for j, (x, y) in enumerate(position(WINDOW_SIZE, w, h)):
            f = {}
            f_3v, v = feature3v(img, x, y, w, h)
            f['name'] = 'feature3v'
            f['position'] = f_3v
            fea3v.append(f)
            value.append(v)
    return fea3v, value


def getfea4(img):
    fea4 = []
    value = []
    for i, (w, h) in enumerate(possible_shapes(2, 2, WINDOW_SIZE)):
        for j, (x, y) in enumerate(position(WINDOW_SIZE, w, h)):
            f = {}
            f_4, v = feature4(img, x, y, w, h)
            f['name'] = 'feature4'
            f['position'] = f_4
            fea4.append(f)
            value.append(v)
    return fea4, value


def findthreshold(data, label, step, D):
    data = data.reshape(-1, 1)
    label = label.reshape(-1, 1)
    colmin = min(data)
    colmax = max(data)
    stepSize = (colmax - colmin) / float(step)
    minerror = np.inf
    for i in range(-1, int(step) + 1):
        tempmatrixlow = np.ones((len(data), 1)).reshape(-1, 1)
        errmatrixlow = np.ones((len(data), 1)).reshape(-1, 1)
        tempmatrixhigh = np.ones((len(data), 1)).reshape(-1, 1)
        errmatrixhigh = np.ones((len(data), 1)).reshape(-1, 1)
        thresh = colmin + float(i) * stepSize
        tempmatrixlow[data >= thresh] = -1
        errmatrixlow[tempmatrixlow == label] = 0
        tempmatrixhigh[data <= thresh] = -1
        errmatrixhigh[tempmatrixlow == label] = 0
        Errorlow = D.T @ errmatrixlow
        Errorhigh = D.T @ errmatrixhigh
        if Errorlow <= Errorhigh:
            if Errorlow < minerror:
                minerror = Errorlow
                threshold = thresh
                polarity = -1
                prelabel = tempmatrixlow
        else:
            if Errorhigh < minerror:
                minerror = Errorhigh
                threshold = thresh
                polarity = 1
                prelabel = tempmatrixhigh

    return prelabel, threshold, polarity, minerror


def findbestfeature(dataset, labelset, D, step):
    m, n = dataset.shape
    # 1:num of feature, 2:polarity(-1:lessthan,1:largerthan 3: threshold
    bestClass = {}
    bestpre = np.zeros((m, 1))
    minError = np.inf
    print("--- Find best feature ---")
    for i in range(n):
        predictlabel, threshold, polarity, error = findthreshold(dataset[:, i], labelset, step, D)
        if error < minError:
            bestpre = predictlabel.copy()
            bestClass['dim'] = i
            bestClass['polarity'] = polarity
            bestClass['thresh'] = threshold
            minError = error
    return bestClass, minError, bestpre


def Train_adaBoost(dataset, labelset, classifierNum, feature):
    weakclass = []
    m = dataset.shape[0]
    D = np.ones((m, 1)) / m
    D = D.reshape(-1, 1)
    prelabel = np.zeros((m, 1))

    #    errormatrix = np.zeros((m, 1))
    for i in range(classifierNum):
        step = 100
        best_class, error, pre_label = findbestfeature(dataset, labelset, D, step)
        a = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        best_class['a'] = a
        best_class['name'] = feature[best_class['dim']]['name']
        best_class['position'] = feature[best_class['dim']]['position']
        weakclass.append(best_class)
        expon = np.multiply(-1 * a * labelset, pre_label)
        expon = np.exp(expon)
        Z = np.sum(D * expon)
        D = D / Z
        print("The error in %d th:%f" % (i, error))
    #       Can shorter training time, but sacrifice the performance
    #        prelabel = prelabel + alpha * pre_label
    #        errormatrix = np.multiply(np.sign(prelabel) != np.mat(labelset), np.ones((m, 1)))
    #        errorRate = np.sum(errormatrix) / m
    #        print("The error after %d th weak classifier: %f" % (i, errorRate))
    #        if errorRate == 0.0:
    #            break
    return weakclass


def find_value(img, name: str, position):
    x = position[0]
    y = position[1]
    w = position[2]
    h = position[3]
    if name == 'feature2h':
        _, val = feature2h(img, x, y, w, h)
    elif name == 'feature2v':
        _, val = feature2v(img, x, y, w, h)
    elif name == 'feature3h':
        _, val = feature3h(img, x, y, w, h)
    elif name == 'feature3v':
        _, val = feature3v(img, x, y, w, h)
    elif name == 'feature4':
        _, val = feature4(img, x, y, w, h)
    return val


def strong_class(image, weakclass, window, face,weight):
    sum = face
    m, n = image.shape
    image = to_integral(image, m, n)
    #    plt.imshow(image)
    for w in weakclass:
        value = find_value(image, w['name'], w['position'])
        if w['polarity'] == '-1':
            if value <= w['thresh']:
                is_face = 1
            else:
                is_face = -1
        else:
            if value >= w['thresh']:
                is_face = 1
            else:
                is_face = -1
        sum += float(w['a']) * is_face
        weight += float(w['a'])
    return sum,weight


def cascade_classifier(image, weakclass, window_size, scale):
    rows, cols = image.shape
    coordinates1 = []
    coordinates2 = []
    coordinates3 = []
    coordinates4 = []
    coordinates5 = []
#    if max(rows,cols)<100:
#        step = 2
#    elif max(rows,cols)<500:
#        step = 5
#    else:
#        step = 10
    if max(rows,cols)<200:
       step=1
    elif max(rows,cols)<600:
        step=3
    else:
        step=5
        

    for row in range(0, rows - window_size + 1, step):
        for col in range(0, cols - window_size + 1, step):
            x1 = row
            x2 = row + window_size
            y1 = col
            y2 = col + window_size
            face = 0
            r=math.floor(row * scale)
            c=math.floor(col * scale)
            w=math.floor(window_size * scale)
            weight=0
            face,weight = strong_class(image[x1:x2, y1:y2], weakclass[:2], window_size, face,weight)
            
            if face < weight/2:
                continue
            coordinates1.append((r, c, w, face))

            face,weight = strong_class(image[x1:x2, y1:y2], weakclass[2:12], window_size, face,weight)
            if face < weight/2:
                continue
            coordinates2.append((r, c, w, face))

            face,weight = strong_class(image[x1:x2, y1:y2], weakclass[12:36], window_size, face,weight)
            if face < weight/2:
                continue
            coordinates3.append((r, c, w, face))

            face,weight = strong_class(image[x1:x2, y1:y2], weakclass[36:87], window_size, face,weight)
            if face < weight/2:
                continue
            coordinates4.append((r, c, w, face))

            face,weight = strong_class(image[x1:x2, y1:y2], weakclass[87:-1], window_size, face,weight)
            if face < weight/2:
                continue
            coordinates5.append((r, c, w, face))
    return coordinates3


def nms(bbox,thresh,step):
    if bbox==[]:
        return []
    boxes = np.array(bbox)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2=boxes[:,2]+boxes[:,0]
    y2=boxes[:,2]+boxes[:,1]
    box_score=boxes[:,3]
    area = np.square(boxes[:,2])
    scores = np.argsort(box_score)
    adjust_list = []
    while len(scores) > 0:
        idx = scores[-1]
        adjust_list.append(bbox[idx])
        
        tempx1=np.zeros(len(scores)-1)+x1[idx]
        left=np.maximum(tempx1, x1[scores[:-1]])
        tempy1=np.zeros(len(scores)-1)+y1[idx]
        top=np.maximum(tempy1, y1[scores[:-1]])
        tempx2=np.zeros(len(scores)-1)+x2[idx]
        right=np.minimum(tempx2, x2[scores[:-1]])
        tempy2=np.zeros(len(scores)-1)+y2[idx]
        bottom=np.minimum(tempy2, y2[scores[:-1]])
        
        w=np.maximum(0,right-left)
        h=np.maximum(0,bottom-top)
        a = w*h
        b=area[scores[:-1]]
        ious = a / (area[idx] + b - a)
        if step==1:
            next = np.where(ious < thresh)
        else:
            next = np.where((ious < thresh)&(a!=b))
        scores = scores[next]
    return adjust_list


def main():
    parent = os.path.dirname(os.path.realpath(__file__))
    modeldir = './Model_Files/600,500.json'
    modelpath = os.path.join(parent, modeldir)
    if not os.path.exists(modelpath):
        # train
        facedir = './Model_Files/dataset/face'
        backdir = './Model_Files/dataset/back'
        facepath = os.path.join(parent, facedir)
        backpath = os.path.join(parent, backdir)
        face_rgb, face_grey = data_load(facepath)
        back_rgb, back_grey = data_load(backpath)
        p = 500
        n = 500  # number of dataset
        face_data = sample_data(face_grey, p)
        back_data = sample_data(back_grey, n)
        label = np.hstack([np.ones((p,)), -np.ones((n,))])
        data = np.vstack((face_data, back_data))
        print("--- integral dataset ---")
        train_data = np.array([to_integral(x, WINDOW_SIZE, WINDOW_SIZE) for x in data])
        print("--- integral done ---")
        value = []
        feature = []
        print("--- get feature value ---")
        for i in range(train_data.shape[0]):
            fea_2h, value_2h = getfea2h(train_data[i])
            fea_2v, value_2v = getfea2v(train_data[i])
            fea_3h, value_3h = getfea3h(train_data[i])
            fea_3v, value_3v = getfea3v(train_data[i])
            fea_4, value_4 = getfea4(train_data[i])
            value.append(value_2h + value_2v + value_3h + value_3v + value_4)
        feature = fea_2h + fea_2v + fea_3h + fea_3v + fea_4
        value = np.array(value)
        label = label.reshape(-1, 1)
        print("--- get feature value done ---")
        print("--- Training Classifiers ---")
        classifierArray = Train_adaBoost(value, label, 50, feature)
        print("--- Training is done ---")

        for w in range(len(classifierArray)):
            classifierArray[w]['dim'] = str(classifierArray[w]['dim'])
            classifierArray[w]['polarity'] = str(classifierArray[w]['polarity'])
            classifierArray[w]['thresh'] = classifierArray[w]['thresh'].tolist()
            classifierArray[w]['a'] = str(classifierArray[w]['a'])

        output_json = modelpath
        with open(output_json, 'w') as f:
            json.dump(classifierArray, f)
        print("--- Model saving is done ---")
    else:
        # test
        f = open(modelpath, encoding='utf-8')
        load_dict = json.load(f)
        if len(sys.argv) != 2:
            print("--- Use this command: \"python stitch.py [data directory]\" ---")
            sys.exit(0)
        data_dir = sys.argv[1]
        if data_dir.startswith("'") and data_dir.endswith("'"):
            data_dir = data_dir[1:len(data_dir) - 1]
        elif data_dir.endswith("'"):
            data_dir = data_dir[:len(data_dir) - 1]
        elif data_dir.startswith("'"):
            data_dir = data_dir[1:]
        if not data_dir.endswith("/"):
            data_dir += "/"
        data_dir = os.path.join(data_dir)
        print("--- Data directory is [" + data_dir + "] ---")
        # parent = os.path.dirname(os.path.realpath(__file__))
#        # testdir = './Model_Files/data'
#        data_dir = './data'
        testpath = os.path.join(parent, data_dir)
        savedir = './Model_Files/result_image'
        result_dic = []
        start = time.time()
        num = 0
        print("--- Detection start ---")
        for filename in os.listdir(testpath):
            if num%10 == 0:
                print(num," image finished")
            image = cv2.imread(testpath + "/" + filename)
            image_rgb = image
            image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            m, n = image_grey.shape
            minscale = 1
            maxscale = math.floor(min(m / WINDOW_SIZE, n / WINDOW_SIZE))
            coordinate = []
            adjust_coord = []
            # print("Original x: %d, y: %d" % (n, m))
            for scale in np.arange(3, maxscale-1 , 0.5):
                y = math.floor(m / scale)
                x = math.floor(n / scale)
                test1 = cv2.resize(image_grey, (x, y), interpolation=cv2.INTER_AREA)
                coord = cascade_classifier(test1, load_dict, WINDOW_SIZE,
                                           scale)  # coord = [(y, x, window, face_score), ...]
                coordinate.append(coord)

            box = []
            for i in coordinate:
                for j in i:
                    box.append([j[0], j[1], j[2], j[3]])
            adjust_coord = nms(box, 0.3,1)
            #            adjust_coord = nms(adjust_coord,0.05)
            face_num=0
            if adjust_coord == []:
                result = {}
                result['iname'] = filename
                result['bbox'] = [0, 0, 0, 0]
            else:

                 for i in adjust_coord:
                     face_num+=1
                     if face_num>10:
                        continue
                     y = i[0]
                     x = i[1]
                     w = i[2]
                     result = {}
                     result['iname'] = filename
                     result['bbox'] = [x + 1, y + 1, w, w]
                     result_dic.append(result)
                     cv2.rectangle(image_rgb, (x + 1, y + 1), (x + w - 1, y + w - 1), (0, 255, 0),
                              thickness=1)  # 用rectangle对图像进行画框

            savepath = os.path.join(parent, savedir)
            cv2.imwrite(savepath + "/" + filename, image_rgb)  # 保存图片
            result_dic.append(result)
            num += 1
        pertime = (time.time() - start) / num
        print("Average Runtime for each image:", pertime, "sec")
        print("--- Detection is done ---")
        output_json = "./results.json"
        output_json=os.path.join(parent, output_json)
        with open(output_json, 'w') as f:
            json.dump(result_dic, f)
        print("--- Detection saving is done ---")


if __name__ == "__main__":
    main()
