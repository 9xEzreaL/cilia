import cv2
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

if __name__=='__main__':
    x,y,img = tif_to_point(img_path="test.tiff")
    x,y = simple_regression(x,y,img)
    nonlinear(x,y)
