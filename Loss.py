import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from sklearn.utils.extmath import cartesian
import math

def crossEntropyLoss():

    ce_loss = tf.keras.losses.BinaryCrossentropy()
    return ce_loss


def weightedCrossEntropyLoss(beta):

    '''beta => value can be used to tune false negatives and false
    positives. E.g; If you want to reduce the number of false
    negatives then set beta > 1, similarly to decrease the number
    of false positives, set bata < 1'''
    def wce_loss(labels, preds):
        preds = tf.convert_to_tensor(preds)
        labels = tf.cast(labels, dtype=preds.dtype)
        b = labels * beta + (1 - labels) * (1 - beta)
        ce = tf.keras.losses.BinaryCrossentropy()
        ce = ce(labels, preds)
        loss = - (b * ce)          
        return K.mean(loss)
    return wce_loss


def focalLoss(alpha=0.25, gamma=2):

    '''
    Here, gamma > 0 and when gamma = 1 Focal Loss works like Cross-
    Entropy loss function
    alpha => range from [0,1]
    '''
    def focal_loss(labels, preds):
        preds = tf.convert_to_tensor(preds)
        preds = tf.cast(preds, dtype=tf.float32)
        labels = tf.cast(labels, dtype=tf.float32)
        preds = K.flatten(preds)        
        labels = K.flatten(labels)

        ce = tf.keras.losses.BinaryCrossentropy()
        ce = ce(labels, preds)
        a = labels * alpha + (1 - labels) * (1 - alpha)
        pt = tf.where(labels == 1, preds, 1 - preds)
        return K.mean(a * (1 - pt) ** gamma * ce)
    return focal_loss


def ioULoss():

    def iou_loss(labels, preds):
        preds = tf.convert_to_tensor(preds)
        labels = tf.cast(labels, dtype=preds.dtype)
        preds = K.flatten(preds)
        labels = K.flatten(labels)
        loss = 1 - tf.math.divide(K.sum(labels * preds, axis=-1) + 1, K.sum(labels + preds - labels * preds, axis=-1) + 1)        
        return K.mean(loss)


def diceLoss():
    
    def dice_loss(labels, preds):
        preds = tf.convert_to_tensor(preds)
        labels = tf.cast(labels, dtype=preds.dtype)        
        preds = K.flatten(preds)
        labels = K.flatten(labels)
        loss = 1 - tf.math.divide(K.sum(2 * labels * preds, axis=-1) + 1, K.sum(labels + preds, axis=-1) + 1)      
        return K.mean(loss)
    return dice_loss

def focalTverskyLoss(beta=0.7, gamma=2.0):

    '''
    Tversky index can also be seen as an generalization
    of Dices coefficient. It adds a weight to FP and FN
    with the help of beta coefficient.
    Similar to Focal Loss, which focuses on hard example
    by down-weighting easy/common ones.
    beta => ranges from [0, 1]
    gamma => ranges from [1,3]
    '''
    def focalTver_loss(labels, preds):
        preds = tf.convert_to_tensor(preds)
        labels = tf.cast(labels, dtype=preds.dtype)  
        tverskyIndex =  tf.math.divide(K.sum(labels * preds) + 1, 
                                       (K.sum(labels * preds) + beta * K.sum((1 - labels) * preds) + (1 - beta) * K.sum(labels * (1 - preds)))+ 1)
        loss = tf.math.pow((1 - tverskyIndex), gamma)
        return K.mean(loss)

    return focalTver_loss


def hausdorffDistanceLoss(w, h, alpha):

    '''
    w => width of images
    h => height of image
    alpha => ranges from [0, 1]
    '''
    all_img_locations = tf.convert_to_tensor(cartesian([np.arange(w), np.arange(h)]), dtype=tf.float32)
    max_dist = math.sqrt(w ** 2 + h ** 2)

    def cdist(A, B):

        na = tf.reduce_sum(tf.square(A), 1)
        nb = tf.reduce_sum(tf.square(B), 1)

        na = tf.reshape(na, [-1, 1])
        nb = tf.reshape(nb, [1, -1])

        D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))
        return D

    def hausD_loss(labels, preds):
        def loss(labels, preds):
            eps = 1e-6
            labels = K.reshape(labels, [w, h])
            gt_points = K.cast(tf.where(labels > 0.5), dtype=tf.float32)
            num_gt_points = tf.shape(gt_points)[0]
            preds = K.flatten(preds)
            p = preds
            p_replicated = tf.squeeze(K.repeat(tf.expand_dims(p, axis=-1), num_gt_points))
            d_matrix = cdist(all_img_locations, gt_points)
            num_est_pts = tf.reduce_sum(p)
            term_1 = (1 / (num_est_pts + eps)) * K.sum(p * K.min(d_matrix, 1))

            d_div_p = K.min((d_matrix + eps) / (p_replicated ** alpha + (eps / max_dist)), 0)
            d_div_p = K.clip(d_div_p, 0, max_dist)
            term_2 = K.mean(d_div_p, axis=0)

            return term_1 + term_2

        batched_losses = tf.map_fn(lambda x:
                                   loss(x[0], x[1]),
                                   (labels, preds),
                                   dtype=tf.float32)
        return K.mean(tf.stack(batched_losses))
    return hausD_loss


def logCoshDiceLoss():
    
    def lcDice_loss(labels, preds):
        dL = diceLoss()
        loss = tf.math.log(tf.math.cosh(dL(labels, preds)))
        return loss

