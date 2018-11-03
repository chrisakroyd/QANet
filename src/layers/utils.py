import tensorflow as tf


def create_mask(lengths, maxlen):
    """ Create a mask of the given length.
        Args:
            lengths: Length of each item [batch_size]
            maxlen: A scalar value for the maximum length within the batch.
        Returns:
            A boolean tensor of shape [batch_size, maxlen]
    """
    return tf.sequence_mask(lengths, maxlen=maxlen)


def apply_mask(inputs, mask, mask_value=-1e12):
    """ Exponential mask for logits.
        During exponent operation 0 becomes 1, therefore instead of multiplying directly by the boolean mask,
        we create a mask which fills False positions with a large negative value. Note: Should always be applied
        before softmax.

         Args:
            inputs: Arbitrary-rank logits tensor to be masked.
            mask: A `boolean` mask tensor.
            mask_value: A scalar value to fill `False` positions.
        Returns:
            Masked inputs with the same shape as `inputs`.
    """
    # Invert mask, fill with mask value and apply to inputs
    return inputs + (1.0 - tf.cast(mask, dtype=tf.float32)) * mask_value


def slice_ops(words, chars, max, char_max):
    """ Slices a word and character tensor to a specified max (sequence length).
        To speed up training, slice the given word + character array to the max within the batch
        so that we operate over a shorter sequence.

        Args:
            words: A tensor of shape [batch_size, ?].
            chars: A tensor of shape [batch_size, ?, ?].
            max: A scalar value for the maximum length within the batch.
            char_max: A scalar value for the max number of chars.
        Returns:
            A word tensor of shape [batch_size, max] and a character tensor of shape [batch_size, max, char_max].
    """
    words = tf.slice(words, begin=(0, 0), size=(-1, max))
    chars = tf.slice(chars, begin=(0, 0, 0), size=(-1, max, char_max))
    return words, chars
