#--*-- coding:utf-8 --*--

import tensorflow as tf
import os

i = 0
filename = ['A.jpg', 'B.jpg', 'C.jpg']
filename_queue = tf.train.string_input_producer(filename, shuffle=False, num_epochs=3)
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

if not os.path.exists('read'):
    os.mkdir('read/')

with tf.Session() as sess:
    tf.local_variables_initializer().run()
    threads = tf.train.start_queue_runners(sess=sess)
    while True:
        i += 1
        image_data = sess.run(value)
        with open('read/test_%d.jpg' %i, 'wb') as f:
            f.write(image_data)