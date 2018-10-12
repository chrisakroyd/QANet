import tensorflow as tf


def mask_logits(inputs, mask, mask_value=-1e30):
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)


def ndim(x):
    dims = x.get_shape()._dims
    return len(dims) if dims is not None else dims


def dot(x, y):
    if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
        x_shape = []
        for i, s in zip(x.get_shape().as_list(), tf.unstack(tf.shape(x))):
            if i is not None:
                x_shape.append(i)
            else:
                x_shape.append(s)
        x_shape = tuple(x_shape)
        y_shape = []
        for i, s in zip(y.get_shape().as_list(), tf.unstack(tf.shape(y))):
            if i is not None:
                y_shape.append(i)
            else:
                y_shape.append(s)
        y_shape = tuple(y_shape)
        y_permute_dim = list(range(ndim(y)))
        y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
        xt = tf.reshape(x, [-1, x_shape[-1]])
        yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
        return tf.reshape(tf.matmul(xt, yt),
                          x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
    # Else shapes are same so simple mat mul
    return tf.matmul(x, y)


def batch_dot(x, y, axes=None):
    if axes is not None:
        adj_x = None if axes[0] == ndim(x) - 1 else True
        adj_y = True if axes[1] == ndim(y) - 1 else None
    else:
        adj_x = None
        adj_y = None
    return tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
