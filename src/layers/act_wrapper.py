import tensorflow as tf
from src import layers


class ACTWrapper(tf.keras.layers.Layer):
    def __init__(self, block=None, hidden_size=128, max_steps=5, act_type='basic', epsilon=0.01, halting_bias_init=1.0,
                 **kwargs):
        """
            Wraps a layer (block) with the Universal Transformer variant of Adaptive Computation Time (ACT).
            Broad idea is that a single layer is repeatedly applied for a maximum of n steps with the network
            learning exactly how many of those steps are required for each example. Reduces both computation time
            and memory.

            Notes:
                Approach in Universal Transformers paper recommends doubling number of layers + units for
                better performance but most benefits are in computation time.

            Related Papers:
            Adaptive Computation Time (Graves, 2016) - https://arxiv.org/pdf/1603.08983v4.pdf
            Universal Transformers (Dehghani et al. 2019)- https://arxiv.org/pdf/1807.03819.pdf

            NOTE: Due to limitations with the while loop this is incompatible with low_memory mode (Gradient
            recomputation)

            Args:
                block: Wrapped block.
                hidden_size: Wrapped block input/output dimensionality.
                max_steps: Maximum ACT steps.
                act_type: ACT variation, one of: basic, global, accumulated.
                epsilon: Acceptable variation betweem units that have/have not halted to be considered halted.
                halting_bias_init: Initialization weight for the ponder bias.
            Returns:
                ACTWrapper instance.
        """
        super(ACTWrapper, self).__init__(**kwargs)

        if act_type not in ['basic', 'global', 'accumulated']:
            raise ValueError('Unknown act type: %s' % act_type)

        self.hidden_size = hidden_size
        self.act_type = act_type
        self.threshold = 1.0 - epsilon
        self.halting_bias_init = halting_bias_init
        self.max_steps = max_steps
        self.epsilon = epsilon

        self.projection = tf.keras.layers.Conv1D(hidden_size,
                                                 kernel_size=1,
                                                 padding='same')

        self.coordinate_embedding = layers.CoordinateEmbedding()
        self.block = block  # We repeatedly apply this single block for max_steps

        if act_type == 'global':
            self.state_slice = slice(0, 1)
        else:
            self.state_slice = slice(0, 2)

    def build(self, input_shape):
        self.rank = len(input_shape)
        self.ponder_weights = self.add_weight(shape=(self.max_steps, self.hidden_size, 1, ),
                                              initializer=tf.initializers.glorot_uniform(),
                                              trainable=True,
                                              name='ponder_weights')

        self.ponder_bias = self.add_weight(shape=(self.max_steps, 1,),
                                           initializer=tf.constant_initializer(self.halting_bias_init),
                                           trainable=True,
                                           name='ponder_bias')
        super(ACTWrapper, self).build(input_shape)

    def act_variables(self, shape):
        """ Generates the zero'd control variables for ACT (halting probability, remainders + n_updated). """
        halting_probability = tf.zeros(shape, name='halting_probability')  # p_t^n
        remainders = tf.zeros(shape, name='remainder')  # R(t)
        n_updates = tf.zeros(shape, name='n_updates')  # N(t)
        return halting_probability, remainders, n_updates

    def call(self, x, training=None, mask=None):
        """ACT based models.

        https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/universal_transformer_util.py#L1022

        Implementations of all act models are based on craffel@'s cl/160711592.
        (1) Basic AUT based on remainder-distribution ACT (position-wise).
        (2) AUT with global halting probability (not position-wise).
        (3) AUT with final state as accumulation of all states.

        Args:
            x: input
            training: Boolean flag for training mode.
            mask: A boolean mask tensor.
        Returns:
            the output tensor,  (ponder_times, remainders)
        """
        # Map down to internal dimensionality if input isn't already in it.
        if not self.hidden_size == x.shape[-1]:
            x = self.projection(x)

        state = x
        state_shape_static = state.get_shape()
        update_shape = tf.shape(state)[self.state_slice]
        # ACT variables.
        halting_probability, remainders, n_updates = self.act_variables(update_shape)
        # Previous cell states (s_t in the paper).
        previous_state = tf.zeros_like(state, name='previous_state')
        # Current step.
        step = 0

        def ut_function(state, step, halting_probability, remainders, n_updates, previous_state):
            """implements act (position-wise halting).
            Args:
              state: 3-D Tensor: [batch_size, length, channel]
              step: indicates number of steps taken so far
              halting_probability: halting probability
              remainders: act remainders
              n_updates: act n_updates
              previous_state: previous state
            Returns:
              transformed_state: transformed state
              step: step+1
              halting_probability: halting probability
              remainders: act remainders
              n_updates: act n_updates
              new_state: new state
            """
            with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
                state = self.coordinate_embedding((state, tf.cast(step, dtype=tf.float32)), training=training)

            with tf.variable_scope('sigmoid_activation_for_pondering'):
                p = tf.tensordot(state, self.ponder_weights[step], [[self.rank - 1], [0]])
                p = p + self.ponder_bias[step]
                p = tf.sigmoid(p)

                if self.act_type == 'global':
                    # average over all positions (as a global halting prob)
                    p = tf.reduce_mean(p, axis=1)
                    p = tf.squeeze(p)
                else:
                    # maintain position-wise probabilities
                    p = tf.squeeze(p, axis=-1)

            # Mask for inputs which have not halted yet
            still_running = tf.cast(tf.less(halting_probability, 1.0), tf.float32)

            # Mask of inputs which halted at this step
            new_halted = tf.cast(
                tf.greater(halting_probability + p * still_running, self.threshold),
                tf.float32) * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = tf.cast(
                tf.less_equal(halting_probability + p * still_running, self.threshold),
                tf.float32) * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability += p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders += new_halted * (1 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability += new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates += still_running + new_halted

            # Compute the weight to be applied to the new state and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = tf.expand_dims(p * still_running + new_halted * remainders, -1)

            if self.act_type == 'global':
                update_weights = tf.expand_dims(update_weights, -1)
            # apply transformation on the state
            transformed_state = self.block(state, training=training, mask=mask)

            # update running part in the weighted state and keep the rest
            new_state = ((transformed_state * update_weights) +
                         (previous_state * (1 - update_weights)))

            if self.act_type == 'accumulated':
                # Add in the weighted state
                new_state = (transformed_state * update_weights) + previous_state

            # remind TensorFlow of everything's shape
            transformed_state.set_shape(state_shape_static)

            for x in [halting_probability, remainders, n_updates]:
                x.set_shape(state_shape_static[self.state_slice])

            new_state.set_shape(state_shape_static)
            step += 1

            return transformed_state, step, halting_probability, remainders, n_updates, new_state

        # While loop stops when this predicate is FALSE.
        # Ie all (probability < 1-eps AND counter < N) are false.
        def should_continue(u0, u1, halting_probability, u2, n_updates, u3):
            del u0, u1, u2, u3
            return tf.reduce_any(
                tf.logical_and(
                    tf.less(halting_probability, self.threshold),
                    tf.less(n_updates, self.max_steps)))

        # Do while loop iterations until predicate above is false.
        _, _, _, remainder, n_updates, new_state = tf.while_loop(
            should_continue, ut_function, (state, step, halting_probability, remainders, n_updates, previous_state),
            maximum_iterations=self.max_steps + 1, swap_memory=True)

        ponder_times = n_updates
        remainders = remainder

        return new_state, (ponder_times, remainders)
