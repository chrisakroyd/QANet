import tensorflow as tf


def create_mask(lengths, maxlen):
    return tf.sequence_mask(lengths, maxlen=maxlen)


def apply_mask(inputs, mask, mask_value=-1e12):
    return inputs + (1.0 - tf.cast(mask, dtype=tf.float32)) * mask_value
