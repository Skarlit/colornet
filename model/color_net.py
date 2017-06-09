import tensorflow as tf
import numpy as np

epsilon = tf.constant(0.00001, dtype=tf.float32)


# http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/data/colorization_sig2016.pdf
class ColorNet:
    def __init__(self):
        self.out = None
        self.loss = None
        self.optimizer = None
        self._is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        self._config = {
            # Low level features network
            'low_1_w': tf.Variable(tf.truncated_normal([3, 3, 1, 64]), name='low_1_w'),
            'low_1_b': tf.Variable(tf.truncated_normal([64]), name='low_1_b'),
            'low_2_w': tf.Variable(tf.truncated_normal([3, 3, 64, 128]), name='low_2_w'),
            'low_2_b': tf.Variable(tf.truncated_normal([128]), name='low_2_b'),

            'low_3_w': tf.Variable(tf.truncated_normal([3, 3, 128, 128]), name='low_3_w'),
            'low_3_b': tf.Variable(tf.truncated_normal([128]), name='low_3_b'),
            'low_4_w': tf.Variable(tf.truncated_normal([3, 3, 128, 256]), name='low_4_w'),
            'low_4_b': tf.Variable(tf.truncated_normal([256]), name='low_4_b'),

            'low_5_w': tf.Variable(tf.truncated_normal([3, 3, 256, 256]), name='low_5_w'),
            'low_5_b': tf.Variable(tf.truncated_normal([256]), name='low_5_b'),
            'low_6_w': tf.Variable(tf.truncated_normal([3, 3, 256, 512]), name='low_6_w'),
            'low_6_b': tf.Variable(tf.truncated_normal([512]), name='low_6_b'),

            # Middle level feature network
            'mid_1_w': tf.Variable(tf.truncated_normal([3, 3, 512, 512]), name='mid_1_w'),
            'mid_1_b': tf.Variable(tf.truncated_normal([512]), name='mid_1_b'),
            'mid_2_w': tf.Variable(tf.truncated_normal([3, 3, 512, 256]), name='mid_2_w'),
            'mid_2_b': tf.Variable(tf.truncated_normal([256]), name='mid_2_b'),

            # Global feature network
               # Conv
            'glo_1_w': tf.Variable(tf.truncated_normal([3, 3, 512, 512]), name='glo_1_w'),
            'glo_1_b': tf.Variable(tf.truncated_normal([512]), name='glo_1_b'),
            'glo_2_w': tf.Variable(tf.truncated_normal([3, 3, 512, 512]), name='glo_2_w'),
            'glo_2_b': tf.Variable(tf.truncated_normal([512]), name='glo_2_b'),
            'glo_3_w': tf.Variable(tf.truncated_normal([3, 3, 512, 512]), name='glo_3_w'),
            'glo_3_b': tf.Variable(tf.truncated_normal([512]), name='glo_3_b'),
            'glo_4_w': tf.Variable(tf.truncated_normal([3, 3, 512, 512]), name='glo_4_w'),
            'glo_4_b': tf.Variable(tf.truncated_normal([512]), name='glo_4_b'),

               # FC  # input, output are reversed.
            'glo_5_w': tf.Variable(tf.truncated_normal([4 * 4 * 512, 1024]), name='glo_5_w'),
            'glo_5_b': tf.Variable(tf.truncated_normal([1024]), name='glo_5_b'),
            'glo_6_w': tf.Variable(tf.truncated_normal([1024, 512]), name='glo_6_w'),
            'glo_6_b': tf.Variable(tf.truncated_normal([512]), name='glo_6_b'),
            'glo_7_w': tf.Variable(tf.truncated_normal([512, 256]), name='glo_7_w'),
            'glo_7_b': tf.Variable(tf.truncated_normal([256]), name='glo_7_b'),

            # Fusion Network
            'fuse_1_w': tf.Variable(tf.truncated_normal([512, 256]), name='fuse_1_w'),
            'fuse_1_b': tf.Variable(tf.truncated_normal([256]), name='fuse_1_b'),

            # Colorization network
            'col_1_w': tf.Variable(tf.truncated_normal([3, 3, 256, 128]), name='col_1_w'),
            'col_1_b': tf.Variable(tf.truncated_normal([128]), name='col_1_b'),

            'col_2_w': tf.Variable(tf.truncated_normal([3, 3, 128, 64]), name='col_2_w'),
            'col_2_b': tf.Variable(tf.truncated_normal([64]), name='col_2_b'),
            'col_3_w': tf.Variable(tf.truncated_normal([3, 3, 64, 64]), name='col_3_w'),
            'col_3_b': tf.Variable(tf.truncated_normal([64]), name='col_3_b'),

            'col_4_w': tf.Variable(tf.truncated_normal([3, 3, 64, 32]), name='col_4_w'),
            'col_4_b': tf.Variable(tf.truncated_normal([32]), name='col_4_b'),
            'col_5_w': tf.Variable(tf.truncated_normal([3, 3, 32, 2]), name='col_5_w'),
            'col_5_b': tf.Variable(tf.truncated_normal([2]), name='col_5_b')
        }

    def _up_sample(self, x):
        return tf.image.resize_images(x, [2 * tf.shape(x)[1], 2 * tf.shape(x)[2]],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def _resize_image(self, image_batch, h, w):
        return tf.image.resize_images(image_batch, [h, w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def _conv_block(self, x, w, b, s, name):
        conv = tf.nn.conv2d(x, filter=w, strides=s, padding="SAME")
        return tf.contrib.layers.batch_norm(tf.nn.bias_add(conv, b),
                                            activation_fn=tf.nn.relu, decay=0.9, is_training=self._is_training)

    def _fc(self, x, w, b, name):
        activation = tf.nn.bias_add(tf.matmul(x, w), b)
        return tf.contrib.layers.batch_norm(activation,
                                            activation_fn=tf.nn.relu, decay=0.9, is_training=self._is_training)

    def _conv_up_sample(self, x, w, name):
        return tf.nn.conv2d_transpose(x, w,
                                      strides=[1, 2, 2, 1], name=name,
                                      output_shape=[tf.shape(x)[0], 2 * tf.shape(x)[1], 2 * tf.shape(x)[2], 2])

    def _lower_feature_net(self, input_image_batch):
        one = [1, 1, 1, 1]
        two = [1, 2, 2, 1]

        config = self._config

        # H
        low_1 = self._conv_block(input_image_batch, config['low_1_w'], config['low_1_b'], two, 'low_1')
        low_2 = self._conv_block(low_1, config['low_2_w'], config['low_2_b'], one, 'low_2')

        # H / 2
        low_3 = self._conv_block(low_2, config['low_3_w'], config['low_3_b'], two, 'low_3')
        low_4 = self._conv_block(low_3, config['low_4_w'], config['low_4_b'], one, 'low_4')

        # H / 4
        low_5 = self._conv_block(low_4, config['low_5_w'], config['low_5_b'], two, 'low_5')
        low_6 = self._conv_block(low_5, config['low_6_w'], config['low_6_b'], one, 'low_6')

        # H / 8
        return low_6

    def _global_feature_net(self, lower_feature_input):
        # input is 28x28 resized images
        one = [1, 1, 1, 1]
        two = [1, 2, 2, 1]
        config = self._config
        glo_1 = self._conv_block(lower_feature_input, config['glo_1_w'], config['glo_1_b'], two, 'glo_1')

        glo_2 = self._conv_block(glo_1, config['glo_2_w'], config['glo_2_b'], one, 'glo_2')

        glo_3 = self._conv_block(glo_2, config['glo_3_w'], config['glo_3_b'], two, 'glo_3')
        glo_4 = self._conv_block(glo_3, config['glo_4_w'], config['glo_4_b'], one, 'glo_4')

        # (batch, 4, 4, 512) -> (batch, 4*4*512)
        glo_5_input = tf.reshape(glo_4, [-1, 4 * 4 * 512])

        glo_5 = self._fc(glo_5_input, config['glo_5_w'], config['glo_5_b'], 'glo_5_fc')

        glo_6 = self._fc(glo_5, config['glo_6_w'], config['glo_6_b'], 'glo_6_fc')

        glo_7 = self._fc(glo_6, config['glo_7_w'], config['glo_7_b'], 'glo_7_fc')

        return glo_7, glo_6

    def _fusion_network(self, mid_feature, global_feature):
        # global_feature shape is [batch, 256]
        # mid_feature is [batch, H/8, W/8, 256 channels]
        config = self._config

        # [batch, 256] -> [batch, 1, 1, 256]
        padded_global = tf.expand_dims(tf.expand_dims(global_feature, 1), 1)

        # [batch, 1, 1, 256] -> [batch, H/8, W/8, 256 channels]
        padded_global = tf.tile(padded_global,
                                [1,  tf.shape(mid_feature)[1], tf.shape(mid_feature)[2], 1])

        # [batch, H/8, W/8, 256] x [batch, H/8, W/8, 256] -> [batch, H/8, W/8, 512]
        fused_input = tf.concat([mid_feature, padded_global], axis=3)

        # [batch, H/8, W/8, 512] -> [batch * H/8 * W/8, 512]
        fused_input = tf.reshape(fused_input, [-1, 512])

        # [batch * H/8 * W/8, 512] x [512, 256] -> [batch * H/8 * W/8, 256]
        fused_activation = tf.matmul(fused_input, config['fuse_1_w'], name="fused_activation")

        # [batch * H/8 * W/8, 256] -> [batch, H/8, W/8, 256]
        fused_activation = tf.reshape(fused_activation, [
            tf.shape(mid_feature)[0],
            tf.shape(mid_feature)[1],
            tf.shape(mid_feature)[2], -1])

        return tf.contrib.layers.batch_norm(
            tf.nn.bias_add(fused_activation, config['fuse_1_b']), activation_fn=tf.nn.relu, is_training=True)

    def build_model(self):
        one = [1, 1, 1, 1]
        config = self._config

        x = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1], name="input")
        # Low-Level Features network
        # H
        low_6 = self._lower_feature_net(x)

        # Mid-Level features network
        # H / 8
        mid_1 = self._conv_block(low_6, config['mid_1_w'], config['mid_1_b'], one, 'mid_1')
        mid_2 = self._conv_block(mid_1, config['mid_2_w'], config['mid_2_b'], one, 'mid_2')

        # Resize Low-level features network
        low_resized_6 = self._lower_feature_net(self._resize_image(x, 112, 112))

        # Global Features network
        glo_7, glo_6 = self._global_feature_net(low_resized_6)

        # Fusion Network
        fused = self._fusion_network(mid_feature=mid_2, global_feature=glo_7)

        #  Colorization network
        # H / 4
        col_1 = self._conv_block(fused, config['col_1_w'], config['col_1_b'], one, 'col_1')

        # H / 2
        col_2 = self._conv_block(self._up_sample(col_1), config['col_2_w'], config['col_2_b'], one, 'col_2')
        col_3 = self._conv_block(col_2, config['col_3_w'], config['col_3_b'], one, 'col_3')

        # H
        col_4 = self._conv_block(self._up_sample(col_3), config['col_4_w'], config['col_4_b'], one, 'col_4')
        col_5 = self._conv_block(col_4, config['col_5_w'], config['col_5_b'], one, 'col_5')

        self.out = tf.identity(self._up_sample(col_5), name="output")

        target = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 2], name="target")
        # TODO: Implement classifier loss for glo_6
        self.loss = tf.reduce_mean(tf.squared_difference(self.out, target), name="loss")
        self.optimizer = tf.train.AdadeltaOptimizer().minimize(self.loss)

if __name__ == "__main__":
    net = ColorNet()
    net.build_model()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        result = sess.run(net.out, feed_dict={'input:0': np.random.random((20, 224, 224, 1)), 'is_training:0': True})
        saver.save(sess, "../tmp/test_model")
        print result.shape
