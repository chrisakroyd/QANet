import tensorflow as tf


def mask_logits(inputs, mask, mask_value=-1e30):
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)
