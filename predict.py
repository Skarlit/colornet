import tensorflow as tf
import numpy as np
import skimage
from skimage import io, transform, color

input_image_path = "./tmp/09.jpg"
checkpoint_path = './tmp/'
model_path = "./model_session.meta"
output_image_path = "./tmp/test_image_out.png"

input_image = io.imread(input_image_path)
input_image = color.rgb2lab(input_image)

image_width = input_image.shape[1]
image_height = input_image.shape[0]

input_L_channel = np.expand_dims(
    np.expand_dims(input_image[:, :, 0], axis=0), axis=-1)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(model_path)
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
    graph = tf.get_default_graph()
    pred = graph.get_tensor_by_name("output:0")
    # TODO: compute loss as well
    ab_channel = sess.run(pred, feed_dict={"input:0": input_L_channel})
    ab_channel = np.squeeze(ab_channel, axis=0)  # get rid of the batch_size dim
    ab_channel = ab_channel[:image_height, :image_width, :]

    L_channel = np.squeeze(input_L_channel, axis=0).astype(dtype=np.float32) # get rid of the batch_size dim
    Lab = np.concatenate((L_channel, ab_channel), axis=2)

    rgb_image = skimage.color.lab2rgb(np.clip(Lab, -1.0, 1.0))
    io.imsave(output_image_path, rgb_image)

    io.imshow(rgb_image)
    io.show()
