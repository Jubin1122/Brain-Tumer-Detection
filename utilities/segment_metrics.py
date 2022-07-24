
from tensorflow.keras import backend as K
import tensorflow as tf

class Quality_metrics:

    def __init__(self, smooth):
        self.smooth = smooth

    def dice_coef(self, y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        intersection = K.sum(y_true * y_pred)
        union = K.sum(y_true) + K.sum(y_pred)
        return (2.0 * intersection + self.smooth) / (union + self.smooth)

    def dice_coef_loss(self,y_true, y_pred):
        return 1 - self.dice_coef(y_true, y_pred)

    def bce_dice_loss(self,y_true, y_pred):
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return self.dice_coef_loss(y_true, y_pred) + bce(y_true, y_pred)

    def iou(self, y_true, y_pred):
        intersection = K.sum(y_true * y_pred)
        sum_ = K.sum(y_true + y_pred)
        jac = (intersection + self.smooth) / (sum_ - intersection + self.smooth)
        return jac