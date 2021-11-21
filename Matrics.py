import tensorflow as tf
import tensorflow.keras.backend as K

def ioU():
    def cal_iou(labels, preds):
        preds = tf.convert_to_tensor(preds)
        labels = tf.cast(labels, dtype=preds.dtype)
        preds = K.flatten(preds)
        labels = K.flatten(labels)
        iou = tf.math.divide(K.sum(labels * preds, axis=-1), K.sum(labels + preds - labels * preds, axis=-1))        
        return K.mean(iou)
    return cal_iou


def dice():
    def cal_dice(labels, preds):
        preds = tf.convert_to_tensor(preds)
        labels = tf.cast(labels, dtype=preds.dtype)
        preds = K.flatten(preds)
        labels = K.flatten(labels)
        dice = tf.math.divide(K.sum(2 * labels * preds, axis=-1) + 1, K.sum(labels + preds, axis=-1) + 1)       
        return K.mean(dice)
    return cal_dice
    

def mAP(threshold=0.5):  
    def cal_map( labels, preds):
        preds = tf.convert_to_tensor(preds)
        labels = tf.cast(labels, dtype=preds.dtype)
        pos = tf.math.reduce_sum(preds, axis=[1,2])
        mask_p = tf.math.greater(pos, 0.00)
        p = len(tf.boolean_mask(pos, mask_p))
        iou = tf.math.divide(tf.math.reduce_sum(labels * preds, axis=[1,2]), 
                             tf.math.reduce_sum(labels + preds - labels*preds, axis=[1,2]))
        mask_iou = tf.math.greater(iou, threshold)
        tp = len(tf.boolean_mask(iou, mask_iou))

        return tp / p
    return cal_map
