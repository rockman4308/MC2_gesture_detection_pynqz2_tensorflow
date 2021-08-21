import sys 
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf


@tf.function
def f(arg):
    tf.print(arg, output_stream=sys.stdout)

@tf.function()
def Index_L1_loss(yTrue,yPred):
    y_true = tf.reshape(yTrue,[-1])
    y_pred = tf.reshape(yPred,[-1])
    y_true_mask = tf.cast(tf.math.not_equal( y_true,tf.constant(0, dtype=tf.float32)),dtype=tf.float32)

    # ret = tf.math.reduce_sum(masked)
    ret = tf.math.reduce_sum(tf.math.abs(y_true-y_pred)* y_true_mask)
    ret = ret / (tf.math.reduce_sum(y_true_mask) + 1e-4)
    # f(ret)
    # f(yTrue_index)
    # AB = tf.math.abs(yTrue-yPred)
    # f(AB)

    return ret

def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        # y_true = y_true[]

        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.sigmoid(y_pred) # sigmoid
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        print(y_pred)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        print(p_t)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed


def focal_loss(hm_true, hm_pred):
    const = 1
    alpha = 2
    beta = 4
    pos_mask = tf.cast(tf.equal(hm_true, 1), tf.float32)
    neg_mask = tf.cast(tf.less(hm_true, 1), tf.float32)
    neg_weights = tf.pow(1 - hm_true, beta)
    
    pos_loss = -tf.math.log(tf.clip_by_value(hm_pred, 1e-4, 1)) * tf.pow(1 - hm_pred, alpha) * pos_mask * const
    neg_loss = -tf.math.log(tf.clip_by_value(1 - hm_pred, 1e-4, 1)) * tf.pow(hm_pred, alpha) * neg_weights * neg_mask

    num_pos = tf.reduce_sum(pos_mask)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)
    

    cls_loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
    return cls_loss




# def FocalLoss(alpha=2,beta=4):
#     def FocalLoss_fix(y_true, y_pred):
#         print("y_true")
#         print(y_true)
#         print("y_pred")
#         print(y_pred)
#         y_pred = tf.clip_by_value(tf.sigmoid(y_pred),clip_value_min=1e-4, clip_value_max=1 - 1e-4 )

#         pos_index = tf.cast( K.equal(y_true,1),dtype=tf.float32)
#         neg_index = tf.cast( K.less(y_true,1),dtype=tf.float32)
        
#         print("pos_index")
#         print(pos_index)
#         print("neg_index")
#         print(neg_index)

#         output = y_pred
#         target = y_true
#         pos_loss = tf.math.pow(1 - output, alpha) * tf.math.log(output) * pos_index
#         neg_loss =  tf.math.pow(1 - target, beta) * tf.math.pow(output, alpha) *  tf.math.log(1 - output) * neg_index

#         pos_loss = K.sum(pos_loss)
#         neg_loss = K.sum(neg_loss)
#         pos_num = K.sum(pos_index)
#         # print(pos_num)
#         loss = 0

#         loss = tf.cond(tf.greater(pos_num,0),lambda :loss - (pos_loss + neg_loss) / pos_num ,lambda: loss - neg_loss )
#         # loss = loss - (pos_loss + neg_loss) / pos_num if pos_num > 0 else loss - neg_loss
#         return loss

#     return FocalLoss_fix

if __name__ == "__main__":

    # data = [[0.         ,0.         ,0.         ,0.         ,0.         ,0.
    #         ,0.         ,0.         ,0.         ,0.         ,0.         ,0.0668072
    #         ,0.07214504 ,0.07780384 ,0.08379332 ,0.09012267 ,0.09680048 ,0.10383468
    #         ,0.11123244 ,0.11900011 ,0.12714315 ,0.13566606 ,0.1445723  ,0.15386423
    #         ,0.16354306 ,0.17360878 ,0.18406013 ,0.19489452 ,0.20610805 ,0.21769544
    #         ,0.22965    ,0.24196365 ,0.25462691 ,0.26762889 ,0.28095731 ,0.29459852
    #         ,0.30853754 ,0.32275811 ,0.33724273 ,0.35197271 ,0.36692826 ,0.38208858
    #         ,0.39743189 ,0.41293558 ,0.42857628 ,0.44433    ,0.46017216 ,0.47607782
    #         ,0.49202169 ,0.50797831 ,0.52392218 ,0.53982784 ,0.55567    ,0.57142372
    #         ,0.58706442 ,0.60256811 ,0.61791142 ,0.63307174 ,0.64802729 ,0.66275727
    #         ,0.67724189 ,0.69146246 ,0.70540148 ,0.71904269 ,0.73237111 ,0.74537309
    #         ,0.75803635 ,0.77035    ,0.78230456 ,0.79389195 ,0.80510548 ,0.81593987
    #         ,0.82639122 ,0.83645694 ,0.84613577 ,0.8554277  ,0.86433394 ,0.87285685
    #         ,0.88099989 ,0.88876756 ,0.89616532 ,0.90319952 ,0.90987733 ,0.91620668
    #         ,0.92219616 ,0.92785496 ,1.         ,0.         ,0.         ,0.
    #         ,0.         ,0.         ,0.         ,0.         ,0.         ,0.
    #         ,0.         ,0.         ,0.         ,0.        ]]
    
    # y_true = np.array([0., 3.,0., 0.],dtype=np.float32)
    # y_pred = np.array([0.6, 4.,0.4, 0.6], dtype=np.float32)

    # b = FocalLoss()

    # loss = b(y_true,y_pred)
    # loss = K.mean(b(y_true,y_pred)).numpy()

    # y_true =  tf.convert_to_tensor( np.reshape(y_true,(4,1)) , dtype=tf.float32)
    # y_pred = tf.convert_to_tensor(  np.reshape(y_pred,(4,1)) , dtype=tf.float32)

    # loss = Index_L1_loss(y_true,y_pred)
    # f(loss)

    Y_true = np.array([[ 0]],dtype=np.float32)
    Y_pred = np.array([[  0.85]], dtype=np.float32)
    print(K.eval(focal_loss(Y_true, Y_pred)))
