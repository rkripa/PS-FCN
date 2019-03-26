import tensorflow as tf
from datetime import datetime

def tensorboard_init(dir="./logs/tensorboard/"):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = dir + current_time + '/train'
    test_log_dir =  dir + current_time + '/test'
    tf_train_writer = tf.summary.create_file_writer(train_log_dir)
    tf_test_writer = tf.summary.create_file_writer(test_log_dir)
    return tf_train_writer, tf_test_writer

def tensorboard_scalar(tf_writer, tag, value, step):
    with tf_writer.as_default():
        tf.summary.scalar(tag, value, step=step)
