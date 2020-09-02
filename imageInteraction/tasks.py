# Create your tasks here
from __future__ import absolute_import, unicode_literals

import copy
from celery import shared_task
from rest_framework.response import Response
import os
from imageDataProcess import FeatureObtainer, DataAugmentation
from imageCoreAlgorithm import ModelTraining
import time
from urllib import parse
from users.models import Users
from imageInteraction.models import ImageModelBasicInfo

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@shared_task
def model_training(request_data, y_dict, images_info):
    # 数据增强、特征提取和模型训练
    time_start = time.time()
    data = request_data.copy()
    for item in data['label']:
        item = parse.unquote(item)
    user_belong = Users.objects.get(username=data['userName'])
    model_name = ImageModelBasicInfo.objects.get(user_belong=user_belong,
                                                 cn_name=data['modelName'], delete_status=0)
    augment_images_info = copy.deepcopy(images_info)
    DataAugmentation.data_augmentation(images_info, augment_images_info)

    if images_info.__len__() < 2:
        return Response("The number of labels must be more than 2!")
    elif images_info.__len__() == 2:
        print("two labels1")
        model_name.train_status = 1
        model_name.algorithm = "Training"
        model_name.save()

        train_acc = ModelTraining.model_training_cnn_binary(os.path.join(BASE_DIR, "aug_images"),
                                                            images_info[0]["model_belong"])
        print("two labels2")
        time_end = time.time()
        model_name.accuracy = round(train_acc, 2)
        model_name.train_time_length = round(time_end - time_start, 1)
        model_name.train_status = 2
        model_name.algorithm = "CNN"
        model_name.save()
        return
    else:
        pre_train_model_type = "VGG16"
        X_train, y_train, X_test, y_test = FeatureObtainer.feature_obtainer(augment_images_info,
                                                                            pre_train_model_type, y_dict)

        model_name.train_status = 1
        model_name.algorithm = "Training"
        model_name.save()

        train_acc = ModelTraining.model_training_cnn_svm(augment_images_info,
                                                         images_info[0]["model_belong"],
                                                         X_train, y_train, X_test, y_test)
        time_end = time.time()
        model_name.accuracy = round(train_acc, 2)
        model_name.train_time_length = round(time_end - time_start, 1)
        model_name.train_status = 2
        model_name.algorithm = "SVM"
        model_name.save()
        return

