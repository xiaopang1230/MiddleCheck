from django.conf.urls import url, include
from imageInteraction import views as image_interaction_views

urlpatterns = [
    # url(r'^techGetImgModel/$', image_classifier_views.ImageClassifierAPI.tech_get_model),
    # url(r'^stuGetImgModel/$', image_classifier_views.ImageClassifierAPI.stu_get_model),
    # url(r'^techGetCoImgModel/$', image_classifier_views.ImageClassifierAPI.teach_get_co_model),
    # url(r'^stuGetCoImgModel/$', image_classifier_views.ImageClassifierAPI.stu_get_co_model),

    url(r'^createImageModel/$', image_interaction_views.ImageClassifierAPI.create_model),
    url(r'^trainImageModel/$', image_interaction_views.ImageClassifierAPI.train_model),
    url(r'^deleteImageModel/$', image_interaction_views.ImageClassifierAPI.delete_model),
    url(r'^testImageModel/$', image_interaction_views.ImageClassifierAPI.test_model),
    url(r'^uploadImg/$', image_interaction_views.ImageClassifierAPI.save_data),
    url(r'^deleteImg/$', image_interaction_views.ImageClassifierAPI.delete_data),
    url(r'^addLabel/$', image_interaction_views.ImageClassifierAPI.save_label),
    url(r'^deleteLabel/$', image_interaction_views.ImageClassifierAPI.delete_label),
    url(r'^imageIfTrain/$', image_interaction_views.ImageClassifierAPI.train_status_check),
    url(r'^editImgModel/$', image_interaction_views.ImageClassifierAPI.edit_img_model),
    url(r'^reTrainImgModel/$', image_interaction_views.ImageClassifierAPI.re_train_img_model),
    url(r'^publishImgModel/$', image_interaction_views.ImageClassifierAPI.publish_img_model),
    url(r'^editStuImgModel/$', image_interaction_views.ImageClassifierAPI.edit_stu_img_model),
    url(r'^stuUploadImg/$', image_interaction_views.ImageClassifierAPI.stu_save_data),
    url(r'^getOutputData/$', image_interaction_views.ImageClassifierAPI.get_output_data),
]