import tensorflow as tf


def ema_ops(train_op, decay):
    with tf.name_scope('ema_ops'):
        ema = tf.train.ExponentialMovingAverage(decay)
        with tf.control_dependencies([train_op]):
            train_op = ema.apply(tf.trainable_variables() + tf.moving_average_variables())
            return train_op, ema


def l2_ops(l2, variables=None):
    if variables is None:
        variables = tf.trainable_variables()

    with tf.name_scope('l2_ops'):
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in variables]) * l2
        return l2_loss


def clip_by_global_norm(gradients, clip_norm=5.0):
    gradients, variables = zip(*gradients)
    clipped_grads, _ = tf.clip_by_global_norm(gradients, clip_norm)
    grads_and_vars = zip(clipped_grads, variables)
    return grads_and_vars


def inverse_exponential_warmup(learn_rate, warmup_steps=1000, global_step=None):
    if global_step is None:
        global_step = tf.train.get_global_step()
    else:
        tf.train.assert_global_step(global_step)

    lr_decay = learn_rate / tf.log(tf.to_float(warmup_steps)) * tf.log(tf.to_float(global_step) + 1)
    return tf.minimum(learn_rate, lr_decay)


def linear_warmup(learn_rate, warmup_steps=1000, global_step=None):
    if global_step is None:
        global_step = tf.train.get_global_step()
    else:
        tf.train.assert_global_step(global_step)

    lr_decay = tf.minimum(1.0, tf.to_float(global_step) / warmup_steps)
    return lr_decay * learn_rate


def lr_warmup(name, learn_rate, warmup_steps=1000, global_step=None):
    if name == 'linear':
        return linear_warmup(learn_rate, warmup_steps, global_step)
    elif name == 'inverse_exp':
        return inverse_exponential_warmup(learn_rate, warmup_steps, global_step)
    else:
        raise ValueError('Invalid warmup scheme, {}.'.format(name))
