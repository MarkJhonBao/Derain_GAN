import tensorflow_graphics as tfg
import tensorflow as tf

from attentive_gan_model.attentive_gan_net import cnn_basenet
from attentive_gan_model.vgg16 import vgg16
from config import global_config

class derain_model(cnn_basenet.CNNBaseModel):
    def __init__(self):
        super(derain_model, self).__init__()
        self._vgg_extractor = vgg16.VGG16Encoder(phase='test')
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

    def _residual_module(self, input_tensor, name='residual_module'):
        with tf.variable_scope(name):
            pass
        pass

    def build_pyramid(self, input_tensor, name='pyramid_module'):
        with tf.variable_scope(name):
            pass

    def BLSTM_module(self, input_tensor, name='BLSTM_moudle'):
        pass

    def activation(self, input_tensor):
        pass

    def inference(self):
        pass

    def compute_loss(self, input_tensor, label_tensor, name, reuse=False):
        pass
