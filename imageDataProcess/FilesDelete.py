import os


def file_delete(file_list):
    if os.path.isfile(file_list):
        os.remove(file_list)
    elif os.path.isdir(file_list):
        label_list = os.listdir(file_list)
        for label in label_list:
            img_list = os.listdir(os.path.join(file_list, label))
            for img in img_list:
                os.remove(os.path.join(os.path.join(file_list, label, img)))
            os.rmdir(os.path.join(file_list, label))
        os.rmdir(file_list)