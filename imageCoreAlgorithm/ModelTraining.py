from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def model_training_cnn_svm(augment_images_info, model_id, X_train, y_train, X_test, y_test):
    if len(augment_images_info) == 2:
        parameters = {'kernel': ['linear'],
                      'C': np.linspace(0.1, 20, 50),
                      'gamma': np.linspace(0.1, 20, 20)}
    else:
        parameters = {'kernel': ['rbf'],
                      'C': np.linspace(0.1, 20, 50),
                      'gamma': np.linspace(0.1, 20, 20)}
    svc = svm.SVC()
    best_svm = GridSearchCV(svc, parameters, cv=5, scoring='accuracy')
    print("SVM model is training !")
    best_svm.fit(X_train, y_train)
    best = best_svm.best_params_
    model = svm.SVC(C=best['C'], kernel='rbf', gamma=best['gamma'])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = metrics.accuracy_score(y_test, y_pred)
    # 模型训练完毕，保存SVM模型
    file_path = os.path.join(BASE_DIR, "image_model")
    is_path_exists = os.path.exists(file_path)
    if not is_path_exists:
        os.makedirs(file_path)
    file_name = model_id + "_svm.m"
    file_path = os.path.join(file_path, file_name)
    joblib.dump(model, file_path)
    print("model training is finished !")
    return y_score


def model_training_cnn_binary(image_path, model_id):
    # 建立模型
    model = Sequential()
    model.add(Conv2D(32, 3, activation='relu', input_shape=(150, 150, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))  # 冗余过大，建议添加一个dropout
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    # 编译
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    # 从图片中直接产生数据和标签
    train_generator = train_datagen.flow_from_directory(os.path.join(image_path, model_id, "train"),
                                                        target_size=(150, 150),
                                                        batch_size=32,
                                                        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(os.path.join(image_path, model_id, "validation"),
                                                            target_size=(150, 150),
                                                            batch_size=32,
                                                            class_mode='binary')
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=50,
                                  epochs=2,
                                  validation_data=validation_generator,
                                  validation_steps=20)

    # 保存整个模型
    file_path = os.path.join(BASE_DIR, "image_model")
    is_path_exists = os.path.exists(file_path)
    if not is_path_exists:
        os.makedirs(file_path)
    file_name = model_id + "_cnn.hdf5"
    file_path = os.path.join(file_path, file_name)
    model.save(file_path)
    return history.history['acc'][-1]


def model_training_mobile_net_svm(augment_images_info, model_id, X_train, y_train, X_test, y_test):
    if len(augment_images_info) == 2:
        parameters = {'kernel': ['linear'],
                      'C': np.linspace(0.1, 20, 50),
                      'gamma': np.linspace(0.1, 20, 20)}
    else:
        parameters = {'kernel': ['rbf'],
                      'C': np.linspace(0.1, 20, 50),
                      'gamma': np.linspace(0.1, 20, 20)}
    svc = svm.SVC()
    best_svm = GridSearchCV(svc, parameters, cv=5, scoring='accuracy')
    print("SVM model is training !")
    best_svm.fit(X_train, y_train)
    best = best_svm.best_params_
    model = svm.SVC(C=best['C'], kernel='rbf', gamma=best['gamma'])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = metrics.accuracy_score(y_test, y_pred)
    # 模型训练完毕，保存SVM模型
    file_path = os.path.join(BASE_DIR, "image_model")
    is_path_exists = os.path.exists(file_path)
    if not is_path_exists:
        os.makedirs(file_path)
    file_name = model_id + "_svm.m"
    file_path = os.path.join(file_path, file_name)
    joblib.dump(model, file_path)
    print("model training is finished !")
    return y_score

