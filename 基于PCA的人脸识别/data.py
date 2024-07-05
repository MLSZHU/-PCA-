import cv2
import numpy as np
import os

# 人脸图像数据采集和预处理
def preprocess_faces(data_folder):
    # 创建空数组用于保存图像数据
    images = []
    f_dict = {}
    # for file in os.listdir(data_folder):
    #     f_dict[file]=file.split('_')[0]
        
    # print(f_dict)
    # sorted_file = dict(sorted(f_dict.items(), key=lambda item: item[1])).keys()
    # 循环遍历数据文件夹中的每个图像文件
    # print(sorted(os.listdir(data_folder)))
    for file_name in os.listdir(data_folder):
        list = []
        # 读取图像文件
        image_path = os.path.join(data_folder, file_name)
        image = cv2.imread(image_path)

        # 将图像转换为灰度图像
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 将图像大小统一为 480×640
        resized_image = cv2.resize(gray_image, (480, 640))
        # cv2.imshow()
        # print(resized_image.shape)
        # for x in range(resized_image.shape[0]):
        #     # 遍历图片每一行的每一列
        #     for y in range(resized_image.shape[1]):
        #         # 将每一处的灰度值添加至列表
        #         list.append(resized_image[x, y])
        

        images.append(resized_image)
    # 将图像列表转换为 Numpy 数组
    images_array = np.array(images)
    return images_array


# 保存处理后的图像数据到文件
def save_data_to_file(data, file_path):
    np.save(file_path, data)
    print("数据已保存到文件:", file_path)

# 设置数据文件夹路径
data_folder = 'training'

# 进行数据预处理
processed_data = preprocess_faces(data_folder)

# 将处理后的数据保存到文件
save_data_to_file(processed_data, 'data.npy/X_training.npy')
