import keras as K
import tensorflow as tf

def reverse_gradient(X, hp_lambda):
    '''Flips the sign of the incoming gradient during training.'''
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @tf.python.framework.ops.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.neg(grad) * hp_lambda]

    g = get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)

    return y

class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, hp_lambda, mask=None):
        return K.reverse_gradient(x, hp_lambda)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items())

