import os
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import gc
import tensorflow as tf
from keras import backend as K
from keras.applications import MobileNet

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def feature_obtainer(augment_images_info, pre_train_model_type, y_dict):
    print("base model of VGG16 is loading!")
    base_model = VGG16(weights='VGGModels/vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                       include_top=True)  # 加载VGG16模型及参数
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('predictions').output)
    print("loading finished!")
    X_list_train = []
    y_train = []
    X_list_test = []
    y_test = []
    for augment_image_info in augment_images_info:
        i = 0
        for item in augment_image_info["images_name"]:
            file_path = os.path.join(BASE_DIR, "aug_images",
                                     augment_image_info["model_belong"],
                                     augment_image_info["label_belong"],
                                     item)
            img = image.load_img(file_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            fc = model.predict(x)  # 获取VGG16/19全连接层特征
            if i <= 0.8 * len(augment_image_info["images_name"]):
                X_list_train.append(fc.tolist()[0])
                y_train.append(y_dict[augment_image_info["label_belong"]])
            else:
                X_list_test.append(fc.tolist()[0])
                y_test.append(y_dict[augment_image_info["label_belong"]])
            i += 1
    X_train = np.array(X_list_train)
    X_test = np.array(X_list_test)
    print("Features has been obtained !")
    K.clear_session()
    tf.reset_default_graph()
    print("memory has been cleared !")
    return X_train, y_train, X_test, y_test


def feature_obtainer_mb(augment_images_info, pre_train_model_type, y_dict):
    print("base model of MobileNet is loading!")
    base_model = MobileNet(weights='imagenet',
                           include_top=True)  # 加载VGG16模型及参数
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('reshape_2').output)
    print("loading finished!")
    X_list_train = []
    y_train = []
    X_list_test = []
    y_test = []
    for augment_image_info in augment_images_info:
        i = 0
        for item in augment_image_info["images_name"]:
            file_path = os.path.join(BASE_DIR, "aug_images",
                                     augment_image_info["model_belong"],
                                     augment_image_info["label_belong"],
                                     item)
            img = image.load_img(file_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            fc = model.predict(x)  # 获取VGG16/19全连接层特征
            if i <= 0.8 * len(augment_image_info["images_name"]):
                X_list_train.append(fc.tolist()[0])
                y_train.append(y_dict[augment_image_info["label_belong"]])
            else:
                X_list_test.append(fc.tolist()[0])
                y_test.append(y_dict[augment_image_info["label_belong"]])
            i += 1
    X_train = np.array(X_list_train)
    X_test = np.array(X_list_test)
    print("Features has been obtained !")
    K.clear_session()
    tf.reset_default_graph()
    print("memory has been cleared !")
    return X_train, y_train, X_test, y_test