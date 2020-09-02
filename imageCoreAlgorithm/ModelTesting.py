import os
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from sklearn.externals import joblib
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import load_img

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def model_testing(test_image, model_id, y_dict, algorithm):
    # print(test_image, type(test_image))
    # 完成对图片的特征提取
    if algorithm == "SVM":
        base_model = VGG16(weights='VGGModels/vgg16_weights_tf_dim_ordering_tf_kernels.h5', include_top=True)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('predictions').output)
        print("test1")
        x = image.load_img(test_image, target_size=(224, 224))
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        fc = model.predict(x)
        # 完成模型的加载和对目标图片的预测
        file_path = os.path.join(BASE_DIR, "image_model", model_id+"_svm.m")
        clf = joblib.load(file_path)
        prediction = clf.predict(fc)
        print(prediction, y_dict)
        K.clear_session()
        tf.reset_default_graph()
        print("memory has been cleared !")
        for item in y_dict:
            if y_dict[item] == prediction[0]:
                print(item)
                return item
        return "error"
    else:
        # 加载权重
        # binary_dict = {}
        # temp_list = os.listdir(os.path.join(BASE_DIR, "aug_images", model_id, "train"))
        # binary_dict[temp_list[0]] = '0'
        # binary_dict[temp_list[1]] = '1'
        model = load_model(os.path.join(BASE_DIR, "image_model", model_id+"_cnn.hdf5"))

        # 加载图像，预测标签
        img = load_img(test_image, target_size=(150, 150))
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        predictions = model.predict_classes(img)
        print(predictions, y_dict)
        K.clear_session()
        tf.reset_default_graph()
        print("memory has been cleared !")
        for item in y_dict:
            if y_dict[item] == str(predictions[0][0]):
                print(item)
                return item
        return "error"