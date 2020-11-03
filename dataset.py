import os

import cv2
import numpy as np


def preprocess_data(path, image_count, image_size):
    print("Загрузка данных...\n")
    list_of_dir = os.listdir(path)
    input_data = []
    target = []
    i = 0

    for directory in list_of_dir:
        for file_name in os.listdir(path + directory):
            i = i + 1
            print(i)
            label = file_name[0] + file_name[1]
            if label[0] == '0':
                target = np.append(target, int(label[1]))
            else:
                target = np.append(target, int(label))
            color_image = cv2.imread(path + directory + '/' + file_name)
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            n_image = np.around(np.divide(gray_image, 255.0), decimals=1)
            input_data = np.append(input_data, n_image)

    input_data = input_data.reshape(image_count, image_size)
    print("\nЗагрузка данных закончена.\n")

    return input_data, target


def test(path, image_size):
    print("Загрузка тестовой картинки...")
    buffer_images = []

    color_image = cv2.imread(path)
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    n_image = np.around(np.divide(gray_image, 255.0), decimals=1)
    buffer_images = np.append(buffer_images, n_image)
    buffer_images = buffer_images.reshape(1, image_size)
    print("Загрузка тестовой картинки закончена.\n")

    return buffer_images
