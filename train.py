import tensorflow as tf
import numpy as np
from model.color_net import ColorNet
from utils import get_files, proc_batch, sample_files


if __name__ == "__main__":
    sample_dir = './tmp/x/'
    sample_size = 2000
    batch_size = 1
    epoch_size = 1000000
    validation_size = 50
    epoch_per_validation = 100
    epoch_per_save = 10000
    model_save_path = 'model_session'
    # TODO: Handle reloading the model from checkpoint
    model_path = None

    files = get_files(sample_dir, sample_size + validation_size)
    validation = proc_batch(files[0:validation_size])

    samples = np.array(files[validation_size:])
    net = ColorNet()
    net.build_model()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        sample_loss = 0
        for epoch in xrange(1, epoch_size):
            sample = proc_batch(sample_files(samples, batch_size))
            _, c = sess.run([net.optimizer, net.loss], feed_dict={
                'input:0': sample[:, :, :, 0:1],
                'target:0': sample[:, :, :, 1:3],
                'is_training:0': True
            })

            sample_loss += c

            if epoch % epoch_per_validation == 0:
                validation_loss = sess.run(net.loss, feed_dict={
                    'input:0': validation[:, :, :, 0:1],
                    'target:0': validation[:, :, :, 1:3],
                    'is_training:0': False
                })
                print "Sample Loss: {0}, Validation Loss: {1}".format(
                    sample_loss / epoch_per_validation, validation_loss)
                sample_loss = 0

                if epoch % epoch_per_save == 0:
                    print "Saving progress"
                    saver.save(sess, model_save_path)

        saver.save(sess, model_save_path)
