import numpy as np
import matplotlib.pyplot as plt

# Load data
X_train = np.load('data.npy/X_training.npy')
n, n_rows, n_columns = X_train.shape

# PCA procedure for eigenfaces
# CODE HERE
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
W=U

# Display the mean individual and eigenfaces as pseudo-images
# Decomment the following lines
fig = plt.figure('mean individual and eigenfaces', figsize=(8, 8))
fig.set_facecolor('white')
colormap = plt.cm.gray
img = mean_face.reshape(n_rows, n_columns)
plt.subplot(int(np.sqrt(n))+1, int(np.sqrt(n))+1, 1)
plt.imshow(img, cmap=colormap)
plt.axis('off')
plt.title('Mean face')

for k in range(1, n+1):
    img = W[:, k-1].reshape(n_rows, n_columns)
    plt.subplot(int(np.sqrt(n))+1, int(np.sqrt(n))+1, k+1)
    plt.imshow(img, cmap=colormap)
    plt.axis('off')
    plt.title('Eigenface {}'.format(k))

plt.savefig('mean_face_and_eigenfaces.png')
plt.show()

# fig = plt.figure('mean individual and eigenfaces', figsize=(8, 8))
# fig.set_facecolor('white')
# colormap = plt.cm.gray
# img = mean_face.reshape(n_rows, n_columns)
# plt.subplot(int(np.sqrt(n))+1, int(np.sqrt(n))+1, 1)
# plt.imshow(img, cmap=colormap)
# plt.axis('off')
# plt.title('Mean face')

# for k in range(1, n+1):
#     img = W[:, k-1].reshape(n_rows, n_columns)
#     # print(img.shape)
#     plt.subplot(int(np.sqrt(n))+1, int(np.sqrt(n))+1, k+1)
#     plt.imshow(img, cmap=colormap)
#     plt.axis('off')
#     plt.title('Eigenface {}'.format(k))

# plt.savefig('mean_face_and_eigenfaces.png')
# plt.show()


# Calculate principal components of the faces in the training set
# CODE HERE
qq=n  #降至多少维，q<=min(图片数量，图片展平后的维数)
principal_components=W[:,:qq]
projected=principal_components.T@X

# Reconstruct the images from these principal components
# CODE HERE
# print('principal_components,projected',principal_components.shape,projected.shape)
X_reconstructed=principal_components@projected

# Display the n reconstructed faces from the training set
# Decomment the following lines
fig = plt.figure('Reconstructed images', figsize=(8, 8))
fig.set_facecolor('white')
colormap = plt.cm.gray

for k in range(n):
    img = X_reconstructed[:,k].reshape(n_rows, n_columns)
    plt.subplot(int(np.sqrt(n))+1, int(np.sqrt(n))+1, k+1)
    plt.imshow(img, cmap=colormap)
    plt.axis('off')
    plt.title('Reconstructed image {}'.format(k+1))

plt.savefig('Reconstructed image.png')
plt.show()

# Calculate the RMSE between original images and reconstructed images
# q stands for the number of principal components
RMSE = []
q_list = []

for q in range(1, n+1):
    q_list.append(q)
    principal_components=W[:,:q]
    projected=principal_components.T@X
    X_reconstructed=principal_components@projected
    RMSE.append((np.sum((X-(X_reconstructed))**2))/(n*n_columns*n_rows))
print(RMSE)

# Plot the RMSE graph against the number q of principal components
# Decomment the following lines
fig = plt.figure('RMSE en fonction du nombre de composantes principales', figsize=(5.5, 5))
fig.set_facecolor('white')
plt.plot(q_list, RMSE, 'r+', markersize=8, linewidth=2)
hx = plt.xlabel('q', fontsize=20)
plt.setp(hx, 'fontsize', 20)
hy = plt.ylabel('RMSE', fontsize=20)
plt.setp(hy, 'fontsize', 20)
plt.savefig('RMSE_vs_q.png')
plt.show()