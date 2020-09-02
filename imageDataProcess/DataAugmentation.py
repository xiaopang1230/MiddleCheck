from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def single_label_augment(augment_factor, image_info, augment_images_info):
    final_augment_factor = augment_factor / len(image_info["images_name"])
    datagen = ImageDataGenerator(rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest')
    raw_path = os.path.join(image_info["base_path"], image_info["model_belong"], image_info["label_belong"])
    aug_path = os.path.join(BASE_DIR, "aug_images", image_info["model_belong"], image_info["label_belong"])
    for item in augment_images_info:
        item["base_path"] = os.path.join(BASE_DIR, "aug_images")
    is_path_exists = os.path.exists(aug_path)
    if not is_path_exists:
        os.makedirs(aug_path)
    for image_name in image_info["images_name"]:
        image_path = raw_path + "/" + str(image_name)
        img = load_img(image_path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=aug_path,
                                  save_prefix=image_info["label_belong"],
                                  save_format='png'):
            i += 1
            if i > final_augment_factor:
                break
    temp_list = os.listdir(aug_path)
    for item in augment_images_info:
        if item["label_belong"] == image_info["label_belong"]:
            item["images_name"] += temp_list
            break


def single_label_augment_binary(image_info):
    datagen = ImageDataGenerator(rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest')
    raw_path = os.path.join(image_info["base_path"], image_info["model_belong"], image_info["label_belong"])
    aug_train_path = os.path.join(BASE_DIR, "aug_images", image_info["model_belong"], "train", image_info["label_belong"])
    aug_validate_path = os.path.join(BASE_DIR, "aug_images", image_info["model_belong"], "validation", image_info["label_belong"])
    is_path_exists = os.path.exists(aug_train_path)
    if not is_path_exists:
        os.makedirs(aug_train_path)
    is_path_exists = os.path.exists(aug_validate_path)
    if not is_path_exists:
        os.makedirs(aug_validate_path)
    for image_name in image_info["images_name"]:
        image_path = raw_path + "/" + str(image_name)
        img = load_img(image_path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=aug_train_path,
                                  save_prefix=image_info["label_belong"],
                                  save_format='png'):
            i += 1
            if i > 48:
                break
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=aug_validate_path,
                                  save_prefix=image_info["label_belong"],
                                  save_format='png'):
            i += 1
            if i > 60:
                break


def data_augmentation(raw_images_info, augment_images_info):
    '''
    :param raw_images_info: 原始数据索引字典
    :return augment_image_info:增强数据索引字典
    '''
    augment_factor = 12
    for item in augment_images_info:
        item["images_name"] = []
    augment_factor *= len(raw_images_info)
    if raw_images_info.__len__() == 2:
        for image_info in raw_images_info:
            single_label_augment_binary(image_info)
    else:
        for image_info in raw_images_info:
            single_label_augment(augment_factor, image_info, augment_images_info)

    # print(raw_images_info)
    # print(augment_images_info)
    # if augment_type == 0:
    #     for image_info in raw_images_info:
    #         images_sum += len(image_info["images_name"])
    #         if images_sum > 600:
    #             print("the data is too big,the principle_0 is not suitable")
    #             augment_factor = 1
    #         else:
    #             augment_factor = 300 / images_sum
    #     for image_info in raw_images_info:
    #         single_label_augment(augment_factor, image_info, augment_images_info)
    # # 以上是增强原则一的增强因子计算方式：600（常量） / 图片总数
    # elif augment_type == 1:
    #     augment_factor = 6
    #     for image_info in raw_images_info:
    #         single_label_augment(augment_factor, image_info, augment_images_info)
    # # 以上是增强原则二的增强因子计算方式：60（常量）
    # elif augment_type == 2:
    #     for image_info in raw_images_info:
    #         if images_sum > 600:
    #             print("the data is big enough,the principle_2 is not suitable")
    #             augment_factor = 1
    #         else:
    #             augment_factor = 600 / len(image_info["images_name"])
    #         single_label_augment(augment_factor, image_info, augment_images_info)
    #         augment_factor = 0
    # # 以上是增强原则三的增强因子计算方式：600（常量） / 每类图片总数
    # elif augment_type == 3:
    #     sum_per_label = len(raw_images_info) * 100
    #     for image_info in raw_images_info:
    #         augment_factor = sum_per_label / len(image_info["images_name"])
    #         single_label_augment(augment_factor, image_info, augment_images_info)
    #         augment_factor = 0
    # 以上是增强原则四的增强因子计算方式：标签数 * 100 / 每类标签下原始图片数量
