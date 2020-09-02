# 本函数完成图像模型数据增强前的所有数据、文件的初始化工作
from urllib import parse
from users.models import Users
from imageInteraction.models import ImageModelBasicInfo, TrainData, LabelMap
from rest_framework.response import Response
import os
from imageDataProcess import FilesDelete
MEDIA_URL = '/media/'


def train_prepare(request_data, BASE_DIR):
    TRAIN_DATA_ROOT = os.path.join(BASE_DIR, "media").replace('\\', '/')  # media即为图片上传的根路径
    AUG_ROOT = os.path.join(BASE_DIR, "aug_images").replace('\\', '/')
    MODEL_ROOT = os.path.join(BASE_DIR, "image_model").replace('\\', '/')
    print(request_data)
    data = request_data.copy()
    for item in data['label']:
        item = parse.unquote(item)
    # 提交标签信息以及训练指令
    user_belong = Users.objects.get(username=data['userName'])
    model_name = ImageModelBasicInfo.objects.get(user_belong=user_belong,
                                                 cn_name=data['modelName'], delete_status=0)
    model_file_name = model_name.en_name
    # if data['isChange'] == 0 and model_name.algorithm != "Untrained":
    #     model_name.public_status = data['publicStatus']
    #     model_name.save()
    #     return Response("model data is same to old version")
    # else:
    # 删除预先保留的数据增强以及模型信息
    if model_name.train_status == 2:
        model_name.accuracy = 0
        model_name.train_time_length = 0
        model_name.save()
        is_path_exists = os.path.exists(os.path.join(AUG_ROOT, model_file_name, "train").replace('\\', '/'))
        if is_path_exists:
            file_path_list = [os.path.join(AUG_ROOT, model_file_name, "train").replace('\\', '/'),
                              os.path.join(AUG_ROOT, model_file_name, "validation").replace('\\', '/'),
                              os.path.join(AUG_ROOT, model_file_name).replace('\\', '/'),
                              os.path.join(MODEL_ROOT, model_file_name + "_cnn.hdf5").replace('\\', '/')]
        else:
            file_path_list = [os.path.join(AUG_ROOT, model_file_name).replace('\\', '/'),
                              os.path.join(MODEL_ROOT, model_file_name + "_svm.m").replace('\\', '/')]
        for filePath in file_path_list:
            FilesDelete.file_delete(filePath)
    # 将标签信息同步至模型基本信息表中
    labels_list = data['label']
    model_name.train_status = 1
    model_name.public_status = data['publicStatus']
    model_name.save()
    # 生成标签对应表，便于训练和测试的统一
    y_dict = {}
    y_dict_map = LabelMap.objects.get(model_name=model_name)
    map_list = y_dict_map.train_labels.split(",")
    label_list = y_dict_map.real_labels.split(",")
    for r_label, t_label in zip(label_list, map_list):
        y_dict.update({r_label: t_label})
    # 生成训练数据信息列表images_info
    images_info = []
    for label in labels_list:
        image_info = {
            "model_belong": ImageModelBasicInfo.objects.get(user_belong=user_belong,
                                                            cn_name=data['modelName'],
                                                            delete_status=0).en_name,
            "label_belong": label,
            "base_path": TRAIN_DATA_ROOT,
            "images_name": []
        }
        images_path = os.path.join(TRAIN_DATA_ROOT, image_info["model_belong"], image_info["label_belong"])
        image_info["images_name"] = os.listdir(images_path)
        logic_delete_list = []
        foreign_key = ImageModelBasicInfo.objects.get(en_name=image_info["model_belong"])
        sql_delete_set = TrainData.objects.filter(model_name=foreign_key,
                                                  label=image_info["label_belong"],
                                                  delete_status=1)
        for item in sql_delete_set:
            logic_delete_list.append(item.image_name)
        image_info["images_name"] = list(set(image_info["images_name"]) - set(logic_delete_list))
        images_info.append(image_info)
    return y_dict, images_info


def re_train_prepare(request_data, BASE_DIR):
    TRAIN_DATA_ROOT = os.path.join(BASE_DIR, "media").replace('\\', '/')  # media即为图片上传的根路径
    AUG_ROOT = os.path.join(BASE_DIR, "aug_images").replace('\\', '/')
    MODEL_ROOT = os.path.join(BASE_DIR, "image_model").replace('\\', '/')
    data = request_data.copy()
    user_belong = Users.objects.get(username=data['userName'])
    model_name = ImageModelBasicInfo.objects.get(user_belong=user_belong,
                                                 cn_name=data['modelName'], delete_status=0)
    # if data['isChange'] == 0 and model_name.algorithm != "Untrained":
    #     model_name.public_status = data['publicStatus']
    #     model_name.save()
    #     return Response("model data is same to old version")
    # else:
    # 删除预先保留的数据增强以及模型信息
    model_name.accuracy = 0
    model_name.train_time_length = 0
    model_name.save()
    model_file_name = model_name.en_name
    is_path_exists = os.path.exists(os.path.join(AUG_ROOT, model_file_name, "train").replace('\\', '/'))
    if is_path_exists:
        file_path_list = [os.path.join(AUG_ROOT, model_file_name, "train").replace('\\', '/'),
                          os.path.join(AUG_ROOT, model_file_name, "validation").replace('\\', '/'),
                          os.path.join(AUG_ROOT, model_file_name).replace('\\', '/'),
                          os.path.join(MODEL_ROOT, model_file_name + "_cnn.hdf5").replace('\\', '/')]
    else:
        file_path_list = [os.path.join(AUG_ROOT, model_file_name).replace('\\', '/'),
                          os.path.join(MODEL_ROOT, model_file_name + "_svm.m").replace('\\', '/')]
    for filePath in file_path_list:
        FilesDelete.file_delete(filePath)
    print("testteste3")
    # 将标签信息同步至模型基本信息表中
    label_map = LabelMap.objects.get(model_name=model_name)
    labels_list = data['label']
    labels_to_save = ""
    maps_to_save = ""
    map_start = 0
    for label in labels_list:
        maps_to_save += str(map_start)
        maps_to_save += ","
        map_start += 1
        labels_to_save += label
        labels_to_save += ","
    labels_to_save = labels_to_save[:-1]
    model_name.labels = labels_to_save
    model_name.train_status = 1
    model_name.public_status =data['publicStatus']
    model_name.save()
    maps_to_save = maps_to_save[:-1]
    label_map.real_labels = labels_to_save
    label_map.train_labels = maps_to_save
    label_map.save()
    # 生成标签对应表，便于训练和测试的统一
    y_dict = {}
    y_dict_map = LabelMap.objects.get(model_name=model_name)
    map_list = y_dict_map.train_labels.split(",")
    label_list = y_dict_map.real_labels.split(",")
    for r_label, t_label in zip(label_list, map_list):
        y_dict.update({r_label: t_label})
    # 生成训练数据信息列表images_info
    images_info = []
    for label in labels_list:
        image_info = {
            "model_belong": ImageModelBasicInfo.objects.get(user_belong=user_belong,
                                                            cn_name=data['modelName'], delete_status=0).en_name,
            "label_belong": label,
            "base_path": TRAIN_DATA_ROOT,
            "images_name": []
        }
        images_path = os.path.join(TRAIN_DATA_ROOT, image_info["model_belong"],
                                   image_info["label_belong"])
        image_info["images_name"] = os.listdir(images_path)
        logic_delete_list = []
        foreign_key = ImageModelBasicInfo.objects.get(en_name=image_info["model_belong"])
        sql_delete_set = TrainData.objects.filter(model_name=foreign_key,
                                                  label=image_info["label_belong"],
                                                  delete_status=1)
        for item in sql_delete_set:
            logic_delete_list.append(item.image_name)
        image_info["images_name"] = list(set(image_info["images_name"]) - set(logic_delete_list))
        images_info.append(image_info)
    return y_dict, images_info