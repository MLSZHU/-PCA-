import numpy as np
import matplotlib.pyplot as plt
import cv2


name = {
    1:"蒋卓璞",
    2:"林远福", 
    3:"彭瑞峰", 
    4:"石祥金", 
    5:"吴萧", 
    6:"俞天宸", 
    7:"张家喻", 
    8:"朱亦晨",
}

def rec_face(image):

    # Load data
    X_train = np.load('data.npy/X_training.npy')
    n, n_rows, n_columns = X_train.shape

    X_flatted=np.zeros((n,n_rows*n_columns))

    #展开图像
    for i in range(n):
        X_flatted[i,:]=X_train[i].flatten()

    #计算平均脸
    mean_face=np.mean(X_flatted,axis=0)

    #训练集去中心化
    X = X_flatted - mean_face
    X=X.T  #去中心化后的展平图像数组，dxn
    print(X.shape)#d×n

    #计算特征值
    covariance = X.T@X #n×n
    _, V = np.linalg.eig(covariance)

    #得到特征空间的基
    U = X@V
    print(U.shape)#d×n
    
    #得到训练集的特征
    X_feature = U.T@X
    print(X_feature.shape)
#测试图像预处理
###################################################
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Test_resized = cv2.resize(gray_image, (480, 640))
    Test_X=Test_resized.flatten()
    Test_X=Test_X-mean_face
    #得到测试集的特征
    X_test = U.T@Test_X
###################################################

#检索匹配
###################################################
    num = 1
    minDistance = float('inf')
    # 遍历 eigen_train_sample 的每一行，在此处，eigen_train_sample.shape[0] = 210。
    for i in range(0, n-1):
        # print(i)
        distance = np.linalg.norm(X_feature[i,:] - X_test)
        print(distance)
        if distance < minDistance:
            minDistance = distance
            # 8个人中，每个人有2张照片，i是记录的第几张照片
            # 因此记录第几个人的num为 i // 2 + 1。
            num = i // 2 + 1
            # print(num)
###################################################

    qq=n  #降至多少维，q<=min(图片数量，图片展平后的维数)
    principal_components=U[:,:qq]
    projected=principal_components.T@Test_X

    X_reconstructed=principal_components@projected


        # 使用 plt.imshow() 展示图片
    # img = X_reconstructed.reshape(n_rows, n_columns)
    # plt.figure(figsize=(8, 6))
    # plt.imshow(img, cmap='gray')
    # plt.axis('off')
    # plt.show()
    return num, X_reconstructed


if __name__ == "__main__":

    image = cv2.imread(r"E:\Desktop\tp3(1)\tp3\validation\4.JPG")
    num,_ = rec_face(image)
    print(name[num])