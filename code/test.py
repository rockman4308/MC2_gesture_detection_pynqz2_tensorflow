# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

from tensorflow.keras.models import load_model
import numpy as np

from loss import focal_loss,Index_L1_loss
from model_post_process import CtDetDecode




class Gesture_Detection():
    def __init__(self,model_name,windows_size=128):
        self.windows_size = windows_size
        # self.model = load_model(f"./saveModel/{model_name}.h5",custom_objects={'focal_loss':focal_loss,'Index_L1_loss':Index_L1_loss})
        self.model = load_model(f"./saveModel/{model_name}",custom_objects={'focal_loss':focal_loss,'Index_L1_loss':Index_L1_loss})
        # self.modelx = load_model(f"./saveModel/{model_name}")
        self.modelx = CtDetDecode(self.model)

    # def predict(self,g_p):
    #     raw_result = self.model.predict(g_p.reshape(1,100,3))
    #     # print(raw_result)
    #     result_hm = list()
    #     result_hm_max_list = list()
    #     maxpool_hm = tf.nn.max_pool1d(input = raw_result[0],ksize = 3, strides=1,padding='SAME')
    #     # print(maxpool_hm)
    #     for hm in maxpool_hm[0].numpy().T:
    #         # print(hm.shape)
    #         # maxpool_hm = tf.nn.max_pool1d(input = hm,ksize = 3, strides=1,padding='SAME')
    #         # print(maxpool_hm)
            
    #         # print(tf.reshape(maxpool_hm[0],[100]))

    #         result_hm.append(hm.reshape(100))
    #         result_hm_max_list.append(np.max(result_hm))

    #     result_hm_max_list_index = np.argmax(result_hm_max_list)
    #     result_hm_max = np.argmax(result_hm[result_hm_max_list_index])
    #     result_hm_max_value = np.max(result_hm[result_hm_max_list_index])

    #     # print(raw_result)
    #     # print(tf.reshape(raw_result[1],[100]).numpy())
    #     # input()
    #     result_wh = tf.reshape(raw_result[1],[100]).numpy()[result_hm_max]

    #     return {'hm':result_hm
    #             ,'hm_max':result_hm_max
    #             ,'hm_max_value':result_hm_max_value
    #             ,'wh':result_wh}

    def predict_2(self,g_p):
        
        g_p = np.array(g_p)/360
        raw_result = self.modelx.predict(g_p.reshape(1,self.windows_size,3))
        # print(raw_result)
        object_result = np.squeeze(raw_result[0])[0]
        # print("raw result")
        # print(raw_result[1])
        # print(raw_result[1].shape)
        # print(object_result)
        result_hm = np.squeeze(raw_result[1].T)
        result_hm_max = object_result[0]
        result_hm_max_value = object_result[2]
        result_wh = int(object_result[1])
        result_classs = int(object_result[3])

        return {'hm':result_hm
                ,'hm_max':result_hm_max
                ,'hm_max_value':result_hm_max_value
                ,'wh':result_wh
                ,'class':result_classs}



    

    
    


