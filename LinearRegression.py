import tensorflow as tf
from tensorflow import Module
import re

class LinearRegressionComponent(Module):
    '''This class is used for initilizing the weights and bias for the (multiple) linear regression'''
    def __init__(self, input_dim):
        ''' The constructor for  the class which sets the weights and bias.
        Args:
      input_dim : input dimension related to the number of features in data
        R'''
        
        self.input_dim = input_dim
        self.weights = tf.Variable(tf.random.normal(shape=(input_dim,1)), name='kernel')
        self.bias = tf.Variable(tf.zeros((1,1))) 
        
        #self.__call__ = self.call
        
    def __call__(self, inputs):
        ''' Computes the forward pass 
        Args:
        inputs: the data
        Returns:
        output: the computed forwar pass '''

        output = tf.matmul(inputs, self.weights) + self.bias
        return output


class LinearRegression(tf.keras.models.Model):
    '''This is class model used for linear regression'''    
    def __init__(self, input_dim):
        '''The constructor for the class which pass input dim to  Linearregression component
        Args:
        input_dim: input dimension related to the number of features in the data
        Returns:
        class object '''
        super(LinearRegression, self).__init__()
        self.regression_component = LinearRegressionComponent(input_dim)

    def call(self, inputs):
        
        output = self.regression_component(inputs)
        return output

