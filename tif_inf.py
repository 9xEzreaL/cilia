import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

# (1984,1538)
# flaot32
# 0~ 78.0

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
def nonlinear(x,y):
    # model = make_pipeline(PolynomialFeatures(23),linear_model.LinearRegression())
    # model.fit(x,y) # not use so far
    x= x.flatten()
    y= y.flatten()
    # model = linear_model.LogisticRegression()
    # model = make_pipeline(PolynomialFeatures(5), linear_model.LinearRegression())
    # model.fit(x,y)

    dic_x = list(x.flatten())
    dic_y = list(y.flatten())
    dic = dict(zip(dic_x,dic_y))

    model = np.poly1d(np.polyfit(x,y,10))
    plt.scatter(x,y,s=1, color='black')
    # x = x.flatten()
    #################
    # temporal use
    x.sort()
    x = x.reshape(-1,1)
    print(x)
    #################
    plt.xlim(0,img.shape[1])
    plt.ylim(img.shape[0],0)
    plt.plot(x, model(x), color='blue', linewidth=1)
    plt.show()

    # length(V), width
    x0 = input("Input starting point(float or int) or control+c to quit: ")
    x1 = input("Input ending point(float or int) or control+c to quit: ")

    x0 = float(x0)
    x1 = float(x1)

    each_width = (x1 - x0)/500
    d = 0
    for i in range(500):
        p0 = model(x0 + i*each_width)
        p1 = model(x0 + (i+1)*each_width)
        d += (((x0 + (i+1)*each_width) - (x0 + i*each_width))**2 + (p1-p0)**2)**0.5

    print("Total length between start point and end point is: ",d," (pixel)")
    return d

if __name__=='__main__':
    x,y,img = tif_to_point(img_path="test1.tiff")
    x,y = simple_regression(x,y,img)
    nonlinear(x,y)
