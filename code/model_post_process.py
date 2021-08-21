
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda

# src: https://www.kaggle.com/diegojohnson/centernet-objects-as-points#III---Model
# use maxpooling as nms
def _nms(heat, kernel=3):
    # hmax = K.pool2d(heat, kernel, padding='same', pool_mode='max')
    hmax = tf.nn.max_pool1d(input = heat,ksize = kernel, strides=1,padding='SAME')
    keep = tf.cast(tf.math.equal(hmax, heat), K.floatx())
    return heat * keep
    
def _ctdet_decode(hm, reg, k=100, output_stride=4):
    hm = _nms(hm)
    hm_shape = tf.shape(hm)
    reg_shape = tf.shape(reg)
    # batch, width, cat = hm_shape[0], hm_shape[2], hm_shape[3]
    batch, width,cat = hm_shape[0], hm_shape[1] ,  hm_shape[2]
    # tf.print(hm_shape)

    hm_flat = tf.reshape(hm, (batch, -1))
    # reg_flat = tf.reshape(reg, (reg_shape[0], -1, reg_shape[-1]))
    reg_flat = tf.reshape(reg, (batch, -1))

    def _process_sample(args):
        _hm, _reg = args
        # tf.print(_hm)
        _scores, _inds_1d = tf.math.top_k(_hm, k=k, sorted=True)
        # tf.print(_scores,_inds_1d,batch)
        _class = tf.math.floormod(_inds_1d , cat )
        _inds = tf.math.floordiv(_inds_1d,cat)
        # tf.print(_inds)
        _class_float = K.cast(_class , tf.float32)
        _inds_float = K.cast(_inds , tf.float32)
        # tf.print(_inds_float)
        _reg = K.gather(_reg, _inds)
        # tf.print(_reg)
        # get yaw, pitch, roll, x, y, z from regression
        # yaw =  _reg[..., 0]
        # pitch =  _reg[..., 1]
        # roll =  _reg[..., 2]
        # x =  _reg[..., 3] * 100
        # y =  _reg[..., 4] * 100
        # z =  _reg[..., 5] * 100

        # _detection = K.stack([yaw, pitch, roll, x, y, z, _scores], -1)
        _detection = K.stack([ _inds_float,_reg,_scores,_class_float], -1)
        return _detection
    
    detections = K.map_fn(_process_sample, [hm_flat, reg_flat], dtype=K.floatx())
    return detections

def CtDetDecode(model, hm_index=0, reg_index=1, k=3, output_stride=1):
    def _decode(args):
        hm, reg = args
        return _ctdet_decode(hm, reg, k=k, output_stride=output_stride)
    output = Lambda(_decode,name='decode_lambda')([model.outputs[i] for i in [hm_index, reg_index]])
    
    # print(output.summary())
    model = Model(inputs=model.inputs, outputs=[output,model.outputs[hm_index]])
    print(model.summary())
    return model