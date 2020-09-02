from users.models import Users
from imageInteraction.models import ImageModelBasicInfo, TrainData, LabelMap
from rest_framework.decorators import api_view
from rest_framework.response import Response
import os
from imageDataProcess import FilesDelete
from imageCoreAlgorithm import ModelTesting
from urllib import parse
import imageInteraction.tasks as task
from imageInteraction.beforeTraining.trainPreparation import train_prepare, re_train_prepare
from django.conf import settings
import gc
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_URL = '/media/'

TRAIN_DATA_ROOT = os.path.join(BASE_DIR, "media").replace('\\', '/')  # media即为图片上传的根路径
AUG_ROOT = os.path.join(BASE_DIR, "aug_images").replace('\\', '/')
MODEL_ROOT = os.path.join(BASE_DIR, "image_model").replace('\\', '/')
ONLINE_URL = "https://mltest.tuopinpin.com"


class ImageClassifierAPI:
    @api_view(['GET', 'POST'])
    def save_data(request, format=None):
        if request.method == 'GET':
            print("GET")
            return Response()

        elif request.method == 'POST':
            print("POST")
            print(request.data)
            data = request.data
            init_list = [TRAIN_DATA_ROOT, AUG_ROOT, MODEL_ROOT]
            for item in init_list:
                is_path_exists = os.path.exists(item)
                if not is_path_exists:
                    os.makedirs(item)
            account = Users.objects.get(username=data['account'])
            temp = ImageModelBasicInfo.objects.get(user_belong=account, cn_name=data['model_name'], delete_status=0)
            total_count = TrainData.objects.all().count()
            img = TrainData(user_belong=account,
                            model_name=temp,
                            image_id=total_count,
                            label=data['label'],
                            delete_status=data['delete'],
                            content=data['image'],
                            image_name=data['image_name'])
            img.save()
            return Response({
                "save_status": "success",
                "image_id": total_count
            })

    @api_view(['GET', 'POST'])
    def delete_data(request, format=None):
        if request.method == 'GET':
            print("GET")
            return Response()

        elif request.method == 'POST':
            print(request.data)
            data = request.data
            print("POST")
            try:
                img = TrainData.objects.get(image_id=data["image_id"])
                img.delete_status = 1
                img.save()
            except Exception as e:
                return Response("logic delete failed")
            return Response("logic delete Success")

    @api_view(['GET', 'POST'])
    def save_label(request, format=None):
        if request.method == 'GET':
            print("GET")
            return Response()

        elif request.method == 'POST':
            print("POST")
            print(request.data)
            data = request.data
            try:
                # save in database
                user_belong = Users.objects.get(username=data['userName'])
                model_name = ImageModelBasicInfo.objects.get(user_belong=user_belong,
                                                             cn_name=data['modelName'], delete_status=0)

                if model_name.labels is not None:
                    print(model_name.labels, len(model_name.labels))
                    if len(model_name.labels) == 0:
                        label_map = LabelMap.objects.get(model_name=model_name)
                        model_name.labels = data['label']
                        label_map.real_labels = data['label']
                        label_map.train_labels = "0"
                    else:
                        label_map = LabelMap.objects.get(model_name=model_name)
                        model_name.labels += ","
                        model_name.labels += data['label']
                        label_map.real_labels = model_name.labels
                        label_map.train_labels += ","
                        label_list = model_name.labels.split(",")
                        label_map.train_labels += str(len(label_list) - 1)
                else:
                    label_map = LabelMap.objects.create(model_name=model_name)
                    model_name.labels = data['label']
                    label_map.real_labels = data['label']
                    label_map.train_labels = "0"

                model_name.save()
                label_map.save()
                return Response("Add Label Success")
            except Exception as e:
                return Response("Add Label Failed")
            return Response("Unknown Error")

    @api_view(['GET', 'POST'])
    def delete_label(request, format=None):
        if request.method == 'GET':
            print("GET")
            return Response()

        elif request.method == 'POST':
            print("POST")
            print(request.data)
            data = request.data
            try:
                # delete in database
                user_belong = Users.objects.get(username=data['userName'])
                model_name = ImageModelBasicInfo.objects.get(user_belong=user_belong, cn_name=data['modelName'],
                                                             delete_status=0)
                label_map = LabelMap.objects.get(model_name=model_name)
                TrainData.objects.filter(model_name=model_name, label=data['label']).delete()
                label_list = model_name.labels.split(",")
                if label_list.count(data['label']) == 1:
                    label_list.remove(data['label'])
                update_labels = ""
                label_numbers = ""
                numbers = 0
                for label in label_list:
                    label_numbers += str(numbers)
                    update_labels += str(label)
                    label_numbers += ","
                    numbers += 1
                    update_labels += ","
                model_name.labels = update_labels[:-1]
                model_name.save()
                label_map.real_labels = update_labels[:-1]
                label_map.train_labels = label_numbers[:-1]
                label_map.save()
                # delete the files
                model_file_name = model_name.en_name
                label_file_name = data['label']
                file_path = os.path.join(TRAIN_DATA_ROOT, model_file_name, label_file_name).replace('\\', '/')
                if os.path.isdir(file_path):
                    delete_list = os.listdir(file_path)
                    while delete_list.__len__() > 0:
                        delete_img = str(file_path) + "/" + str(delete_list[0])
                        os.remove(delete_img)
                        delete_list.remove(delete_list[0])
                    os.rmdir(file_path)
            except Exception as e:
                return Response("Delete Label Failed")
            return Response("Delete Label Success")

    @api_view(['GET', 'POST'])
    def create_model(request, format=None):
        if request.method == 'GET':
            print("GET")
            return Response()

        elif request.method == 'POST':
            print("POST")
            print(request.data)
            data = request.data
            try:
                user_belong = Users.objects.get(username=data['userName'])
                image_model = ImageModelBasicInfo(user_belong=user_belong,
                                                  cn_name=data['modelName'],
                                                  role=data['role'],
                                                  format_class=data['format_class'],
                                                  delete_status=0,
                                                  train_status=0,
                                                  model_type=data['modelType'])
                count = ImageModelBasicInfo.objects.all().count()
                image_model.en_name = "model" + str(count)
                model_name_check = ImageModelBasicInfo.objects.get(user_belong=user_belong,
                                                                   cn_name=data['modelName'], delete_status=0)
            except ImageModelBasicInfo.DoesNotExist:
                image_model.save()
                return Response("Create Image Model Success!")
            else:
                return Response("Model Name Check Failed!")

    @api_view(['GET', 'POST'])
    def delete_model(request, format=None):
        if request.method == 'GET':
            print("GET")
            return Response()

        elif request.method == 'POST':
            print("POST")
            print(request.data)
            data = request.data
            try:
                # delete in database
                user_belong = Users.objects.get(username=data['userName'])
                model_name = ImageModelBasicInfo.objects.get(user_belong=user_belong,
                                                             cn_name=data['modelName'],
                                                             delete_status=0)
                TrainData.objects.filter(user_belong=user_belong, model_name=model_name).delete()
                LabelMap.objects.filter(model_name=model_name).delete()
                ImageModelBasicInfo.objects.filter(user_belong=user_belong,
                                                   cn_name=data['modelName']).delete()
                # print("test")
                # delete the files
                model_file_name = model_name.en_name
                file_path_list = [os.path.join(TRAIN_DATA_ROOT, model_file_name).replace('\\', '/'),
                                  os.path.join(AUG_ROOT, model_file_name).replace('\\', '/'),
                                  os.path.join(MODEL_ROOT, model_file_name + "_svm.m").replace('\\', '/')]
                for filePath in file_path_list:
                    FilesDelete.file_delete(filePath)
                return Response("Delete Model Success")
            except Exception as e:
                return Response("Delete Model Failed")
            return Response("Unknown Error of Model Deleting")

    @api_view(['GET', 'POST'])
    def train_model(request, format=None):
        if request.method == 'GET':
            print("GET")
            return Response()

        elif request.method == 'POST':
            print("POST")
            data = request.data
            print(data)
            user_belong = Users.objects.get(username=data['userName'])
            model_name = ImageModelBasicInfo.objects.get(user_belong=user_belong,
                                                         cn_name=data['modelName'], delete_status=0)
            if data['isChange'] == 0 and model_name.algorithm != "Untrained":
                model_name.public_status = data['publicStatus']
                model_name.save()
                return Response("model data is same to old version")
            else:
                y_dict, images_info = train_prepare(request.data, BASE_DIR)
                train_worker = task.model_training.delay(request.data, y_dict, images_info)
                return Response({
                    "train_result": "Start Training",
                    "worker_id": train_worker.id
                })

    @api_view(['GET', 'POST'])
    def get_output_data(request, format=None):
        if request.method == 'GET':
            print("GET")

        elif request.method == 'POST':
            print("POST")
            data = request.data
            user_belong = Users.objects.get(username=data['userName'])
            model_output = ImageModelBasicInfo.objects.get(user_belong=user_belong, cn_name=data['modelName'])
            return Response({
                "train_status": model_output.train_status,
                "train_time": model_output.train_time_length,
                "train_acc": model_output.accuracy
            })

    @api_view(['GET', 'POST'])
    def test_model(request, format=None):
        if request.method == 'GET':
            print("GET")
            return Response()

        elif request.method == 'POST':
            print("POST")
            print(request.data)
            # 提交测试图片以及测试指令
            user_belong = Users.objects.get(username=request.data.get('account'))
            model_info_update = ImageModelBasicInfo.objects.get(user_belong=user_belong,
                                                                cn_name=request.data.get('modelName'),
                                                                delete_status=0)
            if model_info_update.train_status == 0:
                return Response("Model is Untrained!")
            elif model_info_update.train_status == 1:
                return Response("Model is Training!")
            else:
                # 生成标签对应表，便于训练和测试的统一
                y_dict = {}
                y_dict_map = LabelMap.objects.get(model_name=model_info_update)
                map_list = y_dict_map.train_labels.split(",")
                label_list = y_dict_map.real_labels.split(",")
                for r_label, t_label in zip(label_list, map_list):
                    y_dict.update({r_label: t_label})
                test_result = ModelTesting.model_testing(request.FILES.get('img'), model_info_update.en_name,
                                                         y_dict, model_info_update.algorithm)
                return Response(test_result)

    @api_view(['GET', 'POST'])
    def train_status_check(request, format=None):
        if request.method == 'GET':
            print("GET")
            return Response()

        elif request.method == 'POST':
            print("POST")
            print(request.data)
            data = request.data
            user_belong = Users.objects.get(username=data['username'])
            train_status = ImageModelBasicInfo.objects.get(user_belong=user_belong,
                                                           cn_name=data['modelName'],
                                                           delete_status=0).train_status
            if train_status == 2:
                return Response("模型已训练")
            elif train_status == 1:
                return Response("模型训练中")
            else:
                return Response("模型未训练")

    @api_view(['GET', 'POST'])
    def edit_img_model(request, format=None):
        if request.method == 'GET':
            print("GET")
            return Response()

        elif request.method == 'POST':
            print("POST")
            print(request.data)
            data = request.data
            user_belong = Users.objects.get(username=data['username'])
            model_info = ImageModelBasicInfo.objects.get(user_belong=user_belong,
                                                         cn_name=data['modelName'],
                                                         delete_status=0)
            table_data = []
            if model_info.labels is not None:
                label_list = model_info.labels.split(",")
            else:
                label_list = []
            for label in label_list:
                label_data = {}
                image_name = []
                contents = []
                image_id = []
                images = TrainData.objects.filter(model_name=model_info, label=label,
                                                  delete_status=0)
                for item in images:
                    image_name.append(item.image_name)
                    # contents.append(os.path.join(settings.LOCAL_URL, parse.unquote(item.content.url)))
                    contents.append(parse.unquote(item.content.url))
                    image_id.append(item.image_id)

                label_data["label"] = label
                label_data["image_name"] = image_name
                label_data["contents"] = contents
                label_data["image_id"] = image_id
                table_data.append(label_data)
                print(model_info.public_status)
            return Response({"tableData": table_data,
                             "publicStatus": model_info.public_status})

    @api_view(['GET', 'POST'])
    def re_train_img_model(request, format=None):
        if request.method == 'GET':
            print("GET")
            return Response()

        elif request.method == 'POST':
            print("POST")
            data = request.data
            print(data)
            user_belong = Users.objects.get(username=data['userName'])
            model_name = ImageModelBasicInfo.objects.get(user_belong=user_belong,
                                                         cn_name=data['modelName'], delete_status=0)
            model_name.public_status = data['publicStatus']
            model_name.save()
            if data['isChange'] == 0 and model_name.algorithm != "Untrained":
                return Response("model data is same to old version")
            else:
                y_dict, images_info = re_train_prepare(data, BASE_DIR)
                print("y_dict", y_dict, images_info)
                train_worker = task.model_training.delay(data, y_dict, images_info)
                print("test3")
                return Response({
                    "train_result": "Start Training",
                    "worker_id": train_worker.id
                })

    @api_view(['GET', 'POST'])
    def publish_img_model(request, format=None):
        if request.method == 'GET':
            print("GET")
            return Response()

        elif request.method == 'POST':
            print("POST")
            print(request.data)
            data = request.data
            try:
                user_belong = Users.objects.get(username=data['userName'])
                model_name = ImageModelBasicInfo.objects.get(user_belong=user_belong,
                                                             cn_name=data['modelName'], delete_status=0)
                model_name.public_status = data['publicStatus']
                model_name.save()
                return Response("Publish Success!")
            except Exception as e:
                return Response("Publish Failed!")

    @api_view(['GET', 'POST'])
    def edit_stu_img_model(request, format=None):
        if request.method == 'GET':
            print("GET")
            return Response()

        elif request.method == 'POST':
            print("POST")
            print(request.data)
            data = request.data
            teacher_belong = Users.objects.get(username=data['teacherName'])
            model_info = ImageModelBasicInfo.objects.get(user_belong=teacher_belong,
                                                         cn_name=data['modelName'],
                                                         delete_status=0)
            student_belong = Users.objects.get(username=data['studentName'])
            table_data = []
            if model_info.labels is not None:
                label_list = model_info.labels.split(",")
            else:
                label_list = []
            for label in label_list:
                label_data = {}
                image_name = []
                contents = []
                image_id = []
                images = TrainData.objects.filter(user_belong=student_belong, model_name=model_info, label=label,
                                                  delete_status=0)
                for item in images:
                    image_name.append(item.image_name)
                    contents.append("http://www.localhost.com:8082" + item.content.url)
                    image_id.append(item.image_id)
                label_data["label"] = label
                label_data["image_name"] = image_name
                label_data["contents"] = contents
                label_data["image_id"] = image_id
                table_data.append(label_data)
            return Response({"tableData": table_data})

    @api_view(['GET', 'POST'])
    def stu_save_data(request, format=None):
        if request.method == 'GET':
            print("GET")
            return Response()

        elif request.method == 'POST':
            print("POST")
            print(request.data)
            data = request.data
            try:
                teacher_belong = Users.objects.get(username=data['model_builder'])
                student_belong = Users.objects.get(username=data['student_name'])
                temp = ImageModelBasicInfo.objects.get(user_belong=teacher_belong, cn_name=data['model_name'],
                                                       delete_status=0)
                total_count = TrainData.objects.all().count()
                img = TrainData(user_belong=student_belong,
                                model_name=temp,
                                image_id=total_count,
                                label=data['label'],
                                delete_status=data['delete'],
                                content=data['image'],
                                image_name=data['image_name'])

                img.save()
                return Response({
                    "save_status": "success",
                    "image_id": total_count
                })
            except Exception as e:
                return Response({"save_status": "failed"})
