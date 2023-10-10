import tensorflow as tf
from tensorflow.keras.losses import Loss
import re

class RigdeLoss(Loss):
    ''' Computese mse with l2 penalty '''
    def __init__(self, weights, lambda_=0.01):
        super(RigdeLoss, self).__init__()
        self.lambda_ = lambda_
        self.weights = [var for var in weights if re.match(r'kernel:\d+', var.name)]

    def call(self, y_truth, y_pred):
        ''' computing RigdeLoss for rigde regression
        Args:
        y_truth: groundtruth data
        y_pred:  model prediction
        weights: model weights for current training step
        '''
        mse_loss = tf.reduce_mean(tf.square(y_truth-y_pred))
        penalty =  self.lambda_ * tf.reduce_sum(tf.convert_to_tensor([tf.square(var) for var in self.weights]))
        total_loss = mse_loss + penalty
        return total_loss
