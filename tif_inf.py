import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

# (1984,1538)
# flaot32
# 0~ 78.0

def rotation(img,angle,center=None):
    (h,w) = img.shape[:2]

    if center is None:
        center = (w/2, h/2)
    R = cv2.getRotationMatrix2D(center, angle=angle, scale=1.0)
    img = cv2.warpAffine(img, R, (w, h))

    return img


def rotate_img(img):
    (h, w) = img.shape  # 讀取圖片大小
    center = (w // 2, h // 2)  # 找到圖片中心

    # 第一個參數旋轉中心，第二個參數旋轉角度(-順時針/+逆時針)，第三個參數縮放比例
    M = cv2.getRotationMatrix2D(center, 90, 1.0)

    # 第三個參數變化後的圖片大小
    img = cv2.warpAffine(img, M, (w, h))

    return img
"""
get every point(x,y) in tiff
return x axis and y axis
"""
def tif_to_point(img_path):
    img = cv2.imread(img_path, 2)
    img = (((img - img.min())/img.max())*255.0)
    _, img = cv2.threshold(img, 1, 255, 0)

    print(img.shape)
    plt.imshow(img)
    plt.show()

    ##############
    while True:
        rot = input("Whether to ratate, if yes, type degree(Ex:45), else type n: ")
        if rot!= 'n':
            img = rotation(img, float(rot), center=None)
            plt.imshow(img)
            plt.show()
        else:
            break
    ##############

    x_axis = []
    y_axis = []
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x][y] != 0.0:
                x_axis.append(y)
                y_axis.append(x)

    return x_axis, y_axis, img


"""
just delete noise point 
"""
def simple_regression(x,y,img):
    X = np.array(x)
    Y = np.array(y)

    regr = linear_model.LinearRegression()
    X = X.reshape(-1,1)
    Y = Y.reshape(-1,1)
    regr.fit(X,Y)
    doc = []
    for x_p in range(len(X)):
        X_p = np.array(X[x_p])
        y_p = regr.predict(X_p.reshape(-1,1))
        dist = (y_p[0][0]-y[x_p])**2

        if dist >=50000:
            doc.append(x_p)
    doc.sort(reverse=True)
    for i in doc:
        x.pop(i)
        y.pop(i)
    x = np.array(x).reshape(-1,1)
    y = np.array(y).reshape(-1,1)

    plt.imshow(img)
    plt.scatter(x,y,s=1,color='black')
    plt.xlim(0,img.shape[1])
    plt.ylim(img.shape[0],0)
    plt.plot(x, regr.predict(x), color='blue', linewidth=1)
    plt.show()
    return x, y

"""
1. fit to nonlinear line 
2. show image and decide which point to start with and end
3. input1=start point
   input2=end point
4. auto calculate total length
"""

def chooselinear(x,y):
    colors = ['green', 'purple', 'gold', 'blue', 'black']
    plt.scatter(x,y,s=1, c='red')
    s_x = []
    for count,degree in enumerate([1,2,3,4,5]):
        s_x.append(list(x))
        s_x = np.array([token for st in s_x for token in s_x])
        Degree = degree*3
        model = np.poly1d(np.polyfit(x,y,Degree))
        s_x.sort()
        s_x = s_x.reshape(-1,1)
        plt.plot(s_x, model(s_x), color=colors[count],label='degree %d' %Degree, linewidth=2)
        s_x = []
    plt.xlim(0,img.shape[1])
    plt.ylim(img.shape[0],0)
    plt.legend(loc=2)
    plt.show()

    choose_deg = int(input('Type degree more suitable: '))
    return choose_deg


def nonlinear(x,y):
    x= x.flatten()
    y= y.flatten()
    # model = linear_model.LogisticRegression()
    # model = make_pipeline(PolynomialFeatures(5), linear_model.LinearRegression())
    # model.fit(x,y)
    # x = x.flatten()

    dic_x = list(x.flatten())
    dic_y = list(y.flatten())
    dic = dict(zip(dic_x,dic_y))

    choose_deg = chooselinear(x,y)
    model = np.poly1d(np.polyfit(x,y,choose_deg))
    plt.scatter(x, y, s=1, color='black')
    # temporal use
    x.sort()
    x = x.reshape(-1,1)
    # print(x)

    plt.xlim(0,img.shape[1])
    plt.ylim(img.shape[0],0)
    plt.plot(x, model(x), color='blue', linewidth=1)
    plt.show()

    # length(V), width
    x0 = input("Input starting point(float or int) or control+c to quit: ")
    x1 = input("Input ending point(float or int) or control+c to quit: ")

    x0 = float(x0)
    x1 = float(x1)

    each_width = (x1 - x0)/1000
    d = 0
    for i in range(1000):
        p0 = model(x0 + i*each_width)
        p1 = model(x0 + (i+1)*each_width)
        d += (((x0 + (i+1)*each_width) - (x0 + i*each_width))**2 + (p1-p0)**2)**0.5

    print("Total length between start point and end point is: ",d," (pixel)")
    return d

def width(wimg_path):
    img = cv2.imread(wimg_path, 2)
    money = cv2.imread('money.png').astype('float32')

    if len(img) < len(img[0]):
        img = np.transpose(img,(1,0))
    money = cv2.resize(money, (img.shape[1],img.shape[1]))
    h = img.shape[0]-money.shape[0]
    w = 0
    money = cv2.copyMakeBorder(money, 0, h, 0, w, cv2.BORDER_CONSTANT, value=0)
    print(img.shape)
    img = cv2.GaussianBlur(img, (1, 17), 7)
    img = (((img - img.min())/img.max())*255.0)
    _, img = cv2.threshold(img, 1, 255, 0)
    # plt.imshow(img)
    # plt.show()
    length = math.ceil(len(img)/20)-2

    space = []
    qq = []
    qq2 = []
    qq3 = []
    qq4 = []
    qq5 = []
    qq6 = []
    qq7 = []
    qq8 = []
    qq9 = []
    qq10 = []
    distance = []
    for i in range(length):

        for j in range(20):
            for k in range(len(img[0])):
                if img[i*20 +j][k]!=0.0:
                    space.append([i*20+j,k])

        for l in range(len(space)):
            if space[l][0]< i*20+2:
                qq.append(space[l][1])

            if space[l][0]>= i*20+2 and space[l][0]< i*20+4:
                qq2.append(space[l][1])

            if space[l][0]>= i*20+4 and space[l][0]< i*20+6:
                qq7.append(space[l][1])

            if space[l][0]>= i*20+6 and space[l][0]< i*20+8:
                qq8.append(space[l][1])

            if space[l][0]>= i*20+8 and space[l][0]< i*20+10:
                qq3.append(space[l][1])

            if space[l][0]> i*20+10 and space[l][0]< i*20+12:
                qq4.append(space[l][1])

            if space[l][0]>= i*20+12 and space[l][0]< i*20+14:
                qq9.append(space[l][1])

            if space[l][0]>= i*20+14 and space[l][0]< i*20+16:
                qq10.append(space[l][1])

            if space[l][0] >= i * 20 + 16 and space[l][0] < i * 20 + 18:
                qq5.append(space[l][1])

            if space[l][0] > i * 20 + 18 and space[l][0] < i * 20 + 20:
                qq6.append(space[l][1])
        qq.sort()
        d=math.ceil(len(qq)/100)-1
        D=math.ceil(len(qq)/100*95)-1

        qq2.sort()
        d2 = math.ceil(len(qq2) / 100) - 1
        D2 = math.ceil(len(qq2) / 100 * 95) - 1

        qq3.sort()
        d3 = math.ceil(len(qq3) / 100) - 1
        D3 = math.ceil(len(qq3) / 100 * 95) - 1

        qq4.sort()
        d4 = math.ceil(len(qq4) / 100) - 1
        D4 = math.ceil(len(qq4) / 100 * 95) - 1

        qq5.sort()
        d5 = math.ceil(len(qq5) / 100) - 1
        D5 = math.ceil(len(qq5) / 100 * 95) - 1

        qq6.sort()
        d6 = math.ceil(len(qq6) / 100) - 1
        D6 = math.ceil(len(qq6) / 100 * 95) - 1

        qq7. sort()
        d7 = math.ceil(len(qq7) / 100) - 1
        D7 = math.ceil(len(qq7) / 100 * 95) - 1

        qq8.sort()
        d8 = math.ceil(len(qq8) / 100) - 1
        D8 = math.ceil(len(qq8) / 100 * 95) - 1

        qq9.sort()
        d9 = math.ceil(len(qq9) / 100) - 1
        D9 = math.ceil(len(qq9) / 100 * 95) - 1

        qq10.sort()
        d10 = math.ceil(len(qq10) / 100) - 1
        D10 = math.ceil(len(qq10) / 100 * 95) - 1

        if d!=-1 and d2!=-1 and d3!=-1 and d4!=-1 and d5!=-1 and d6!=-1 and d7!=-1 and d8!=-1 and d9!=-1 and d10!=-1:
            dist = qq[D]-qq[d]
            dist2 = qq2[D2]-qq2[d2]
            dist3 = qq3[D3]-qq3[d3]
            dist4 = qq4[D4]-qq4[d4]
            dist5 = qq5[D5]-qq5[d5]
            dist6 = qq6[D6]-qq6[d6]
            dist7 = qq7[D7]-qq7[d7]
            dist8 = qq8[D8]-qq8[d8]
            dist9 = qq9[D9]-qq9[d9]
            dist10 = qq10[D10]-qq10[d10]

            Dist = (dist+dist2)/2
            Dist2 = (dist3+dist4)/2
            Dist3 = (dist5+dist6)/2
            Dist4 = (dist7+dist8)/2
            Dist5 = (dist9+dist10)/2

            distance.append([i*20+2,Dist,math.ceil((qq[d]+qq2[d2])/2),math.ceil((qq[D]+qq2[D2])/2)])
            distance.append([i*20+6, Dist4, math.ceil((qq7[d7] +qq8[d8])/2), math.ceil((qq7[D7] + qq8[D8])/2)])
            distance.append([i*20+10,Dist2,math.ceil((qq3[d3]+qq4[d4])/2),math.ceil((qq3[D3]+qq4[D4])/2)])
            distance.append([i*20+14, Dist5, math.ceil((qq9[d9]+qq10[d10])/2),math.ceil((qq9[D9]+qq10[D10])/2)])
            distance.append([i*20+18, Dist3, math.ceil((qq5[d5] + qq6[d6]) / 2), math.ceil((qq5[D5] + qq6[D6])/2)])
            qq = []
            qq2 = []
            qq3 = []
            qq4 = []
            qq5 = []
            qq6 = []
            qq7 = []
            qq8 = []
            qq9 = []
            qq10 = []
            space=[]


    new_distance = []
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for i in range(len(distance)):
        cv2.line(img, (distance[i][2], distance[i][0]), (distance[i][3], distance[i][0]), (255,0,255), 3)
        new_distance.append((distance[i][0], distance[i][1]))
    print("(pixel location, distance)",new_distance)
    # plt.imshow(img)
    # plt.show()
    # moneyy = cv2.addWeighted(money, 0.5, img, 0.5, 0)
    # cv2.imshow('My Image', moneyy)
    cv2.imshow('My Image', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__=='__main__':
    # x,y,img = tif_to_point(img_path="test.tiff")
    # x,y = simple_regression(x,y,img)
    # nonlinear(x,y)
    width(wimg_path="testqq.tiff")