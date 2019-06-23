import math
import warnings
import tensorflow as tf
from src import constants, optimizers


def ema_ops(train_op, decay, variables=None):
    """ Adds ops to calculate and maintain Exponential moving average over all trainable weights.
        Args:
            train_op: Result of optimizer.minimize or optimizer.apply_gradients
            decay: Exponential decay rate.
            variables: List of variables to maintain average over.
        Returns:
            Modified train_op and ema object.
    """
    if variables is None:
        variables = tf.trainable_variables()

    with tf.name_scope('ema_ops'):
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        maintain_avg_op = ema.apply(variables)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(maintain_avg_op)
            return train_op, ema


def l2_ops(l2, variables=None):
    """ Adds ops to calculate l2 loss over all trainable variables.
        Args:
            l2: L2 penalty to apply.
            variables: A list of trainable variables, if None, uses tf.trainable_variables()
        Returns:
            Tensor for the sum of l2_loss over all given variables.
    """
    if variables is None:
        variables = tf.trainable_variables()

    with tf.name_scope('l2_ops'):
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in variables]) * l2
        return l2_loss


def clip_by_global_norm(gradients, clip_norm=5.0):
    """ Applies clipping by global norm to a list of gradient tensors.
        Args:
            gradients: list of gradient tensors.
            clip_norm: Norm of the gradient value to clip by.
        Returns:
            Zipped gradients and variables.
    """
    gradients, variables = zip(*gradients)
    clipped_grads, _ = tf.clip_by_global_norm(gradients, clip_norm)
    grads_and_vars = zip(clipped_grads, variables)
    return grads_and_vars


def inverse_exponential_warmup(learn_rate, warmup_steps=1000, global_step=None):
    """ Learning rate warmup with an inverse exponential curve, after warmup_steps returns learn_rate.
        Args:
            learn_rate: Learn rate.
            warmup_steps: Number of steps till we reach the given learn_rate.
            global_step: Global step variable.
        Returns:
            An op to calculate the learn rate at a given step.
    """
    if global_step is None:
        global_step = tf.train.get_global_step()

    if warmup_steps < 0:
        raise ValueError(constants.ErrorMessages.INVALID_WARMUP_STEPS.format(steps=warmup_steps))
    elif warmup_steps == 0:
        return learn_rate

    lr_decay = learn_rate / tf.log(tf.cast(warmup_steps, dtype=tf.float32)) * tf.log(tf.cast(global_step, dtype=tf.float32) + 1)
    return tf.minimum(learn_rate, lr_decay)


def linear_warmup(learn_rate, warmup_steps=1000, global_step=None):
    """ Learning rate warmup at a linear rate, after warmup_steps returns learn_rate.
        Args:
            learn_rate: Learn rate.
            warmup_steps: Number of steps till we reach the given learn_rate.
            global_step: Global step variable.
        Returns:
            An op to calculate the learn rate at a given step.
    """
    if global_step is None:
        global_step = tf.train.get_global_step()

    if warmup_steps < 0:
        raise ValueError(constants.ErrorMessages.INVALID_WARMUP_STEPS.format(steps=warmup_steps))
    elif warmup_steps == 0:
        return learn_rate

    lr_decay = tf.minimum(1.0, tf.cast(global_step, dtype=tf.float32) / warmup_steps)
    return lr_decay * learn_rate


def get_warmup_scheme(name, learn_rate, warmup_steps=1000, global_step=None):
    """ Gets and returns the named warmup scheme, if name is None just returns the learn_rate.
        Args:
            name: Name of the warmup scheme, either linear or inverse_exp otherwise simply returns learn rate.
            learn_rate: Base learn rate.
            warmup_steps: Number of steps till we reach the given learn_rate.
            global_step: Global step variable.
        Returns:
            An op to calculate the learn rate at a given step.
    """
    if warmup_steps < 0:
        raise ValueError(constants.ErrorMessages.INVALID_WARMUP_STEPS.format(steps=warmup_steps))
    elif warmup_steps == 0:
        return learn_rate

    name = name.lower().strip()
    if name == 'linear':
        return linear_warmup(learn_rate, warmup_steps, global_step)
    elif name == 'inverse_exp':
        return inverse_exponential_warmup(learn_rate, warmup_steps, global_step)

    if name is not None:
        warnings.warn('Invalid warmup scheme specified, using learn rate.')
    return learn_rate


def cosine_decay_with_warmup(learn_rate, total_steps, warmup_scheme='linear',
                             warmup_steps=0, plateau_steps=0, global_step=None):
    """ Cosine decay schedule with a warmup and plateau period.

        Cosine annealing learning rate as described in: SGDR: Stochastic Gradient Descent with Warm Restarts
        (https://arxiv.org/abs/1608.03983). We apply the warmup_scheme for warmup_steps before holding that rate for
        plateau_steps, after plateau_steps + warmup_steps we apply cosine decay for the remaining steps.

        Args:
            learn_rate: The base learn rate.
            total_steps: Total steps used for training.
            warmup_scheme: The name of the warmup scheme to use, if unspecified uses none.
            warmup_steps: The number of steps till we reach the given learn_rate.
            plateau_steps: The number of steps after warmup to use learn_rate for.
            global_step: The current global step, if none retrieves it using tf.train.get_global_step().
        Returns:
            A tensorflow op that calculates the learning rate for the current step.
    """
    if global_step is None:
        global_step = tf.train.get_global_step()

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to warmup_steps.')

    start_step = global_step - warmup_steps - plateau_steps
    decay_steps = total_steps - warmup_steps - plateau_steps
    cosine_decay = 0.5 * (1.0 + tf.cos(tf.constant(math.pi, dtype=tf.float32) * tf.cast(start_step, dtype=tf.float32)
                                       / float(decay_steps)))
    decayed_rate = learn_rate * cosine_decay

    if warmup_steps > 0:
        warmup_rate = get_warmup_scheme(warmup_scheme, learn_rate, warmup_steps, global_step)
        decayed_rate = tf.cond(global_step < warmup_steps + plateau_steps, lambda: warmup_rate, lambda: decayed_rate)
    elif plateau_steps > 0:
        decayed_rate = tf.cond(global_step > plateau_steps, lambda: decayed_rate, lambda: learn_rate)

    return decayed_rate


def construct_train_op(loss, learn_rate=0.001, warmup_scheme='inverse_exp', warmup_steps=1000, plateau_steps=0,
                       total_steps=0, use_cosine_decay=False, clip_norm=5.0, ema_decay=0.999, beta1=0.8, beta2=0.999,
                       epsilon=1e-7, optimizer='adam', weight_decay=0.00001, global_step=None):
    """ Constructs a training op with options for warmup schemes, gradient clipping,
        exponential moving average weight decay + learn rate decay.

        Args:
            loss: A tensor full of loss values.
            learn_rate: Learn rate.
            warmup_scheme: Name of the warmup scheme to utilise, if None or invalid just uses learn_rate.
            warmup_steps: Number of steps till we reach the given learn_rate.
            plateau_steps: The number of steps after warmup to use learn_rate for.
            total_steps: Total steps used for training.
            use_cosine_decay: Optionally apply cosine decay to the learn_rate.
            clip_norm: A scalar float.
            ema_decay: Decay rate to use.
            beta1: beta1 param for the adam optimizer.
            beta2: beta2 param for the adam optimizer.
            epsilon: A small constant for numerical stability.
            optimizer: Optimizer to use.
            weight_decay: Weight decay parameter for AdamW optimizer.
            global_step: Global step variable.
        Returns:
            A train op.
    """
    if global_step is None:
        global_step = tf.train.get_global_step()

    if use_cosine_decay:
        if total_steps == 0:
            raise ValueError('total_steps must be non-zero when using cosine decay.')

        learning_rate = cosine_decay_with_warmup(learn_rate, total_steps, warmup_scheme=warmup_scheme,
                                                 warmup_steps=warmup_steps, plateau_steps=plateau_steps,
                                                 global_step=global_step)
    else:
        learning_rate = get_warmup_scheme(warmup_scheme, learn_rate, warmup_steps, global_step)

    if optimizer == constants.Optimizers.ADAM:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    elif optimizer == constants.Optimizers.ADAMW:
        optimizer = optimizers.AdamWeightDecayOptimizer(learning_rate=learning_rate,
                                                        weight_decay_rate=weight_decay, beta1=beta1, beta2=beta2,
                                                        epsilon=epsilon, exclude_from_weight_decay=['bias', 'scale'])
    else:
        raise ValueError('Unsupported optimizer.')

    if clip_norm > 0.0:
        grads = optimizer.compute_gradients(loss)
        grads_and_vars = clip_by_global_norm(grads, clip_norm)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    else:
        train_op = optimizer.minimize(loss, global_step=global_step)

    if 0.0 < ema_decay < 1.0:
        train_op, _ = ema_ops(train_op, ema_decay)

    return train_op


def get_saver(ema_decay=0.0, ema_vars_only=False):
    """
        Args:
            ema_decay: Ema decay value.
            ema_vars_only: Boolean flag for restoring EMA variables only.
        Returns:
            A tuple of input tensors.
    """
    if 0.0 < ema_decay < 1.0 and ema_vars_only:
        variable_averages = tf.train.ExponentialMovingAverage(0.)
        return tf.train.Saver(variable_averages.variables_to_restore())

    return tf.train.Saver()
