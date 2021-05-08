# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file defines the decoder"""

import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops

# Note: this function is based on tf.contrib.legacy_seq2seq_attention_decoder, which is now outdated.
# In the future, it would make more sense to write variants on the attention mechanism using the new seq2seq library for tensorflow 1.0: https://www.tensorflow.org/api_guides/python/contrib.seq2seq#Attention
def attention_decoder(decoder_inputs, initial_state, question_states, review_states, sent_states, num, q_padding_mask, r_padding_mask, ds_padding_mask, cell, initial_state_attention=False, pointer_gen=True, use_coverage=False, q_prev_coverage=None, r_prev_coverage=None):
  """
  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor [batch_size x cell.state_size].
    question_states: 3D Tensor [batch_size x q_attn_length x attn_size].
    review_states: 3D Tensor [batch_size x review_num*r_attn_length x attn_size].
    q_padding_mask: 2D Tensor [batch_size x q_attn_length] containing 1s and 0s; indicates which of the question locations are padding (0) or a real token (1).
    r_padding_mask: 3D Tensor [batch_size x review_num x r_attn_length] containing 1s and 0s; indicates which of the question locations are padding (0) or a real token (1).
    cell: rnn_cell.RNNCell defining the cell function and size.
    initial_state_attention:
      Note that this attention decoder passes each decoder input through a linear layer with the previous step's context vector to get a modified version of the input. If initial_state_attention is False, on the first decoder step the "previous context vector" is just a zero vector. If initial_state_attention is True, we use initial_state to (re)calculate the previous step's context vector. We set this to False for train/eval mode (because we call attention_decoder once for all decoder steps) and True for decode mode (because we call attention_decoder once for each decoder step).
    pointer_gen: boolean. If True, calculate the generation probability p_gen for each decoder step.
    use_coverage: boolean. If True, use coverage mechanism.
    prev_coverage:
      If not None, a tensor with shape (batch_size, attn_length). The previous step's coverage vector. This is only not None in decode mode when using coverage.

  Returns:
    outputs: A list of the same length as decoder_inputs of 2D Tensors of
      shape [batch_size x cell.output_size]. The output vectors.
    state: The final state of the decoder. A tensor shape [batch_size x cell.state_size].
    attn_dists: A list containing tensors of shape (batch_size,attn_length).
      The attention distributions for each decoder step.
    p_gens: List of scalars. The values of p_gen for each decoder step. Empty list if pointer_gen=False.
    coverage: Coverage vector on the last step computed. None if use_coverage=False.
  """
  with variable_scope.variable_scope("attention_decoder") as scope:
    batch_size = question_states.get_shape()[0].value # if this line fails, it's because the batch size isn't defined
    attn_size = question_states.get_shape()[2].value # if this line fails, it's because the attention length isn't defined
    s_attn_size = sent_states.get_shape()[2].value
    r_padding_mask = tf.reshape(r_padding_mask, [batch_size, -1])

    # Reshape encoder_states (need to insert a dim)
    question_states = tf.expand_dims(question_states, axis=2) # now is shape (batch_size, attn_len, 1, attn_size)
    review_states = tf.expand_dims(review_states, axis=2)
    sent_states = tf.expand_dims(sent_states, axis=2)

    # To calculate attention, we calculate
    #   v^T tanh(W_h h_i + W_s s_t  + b_attn)
    # where h_i is an encoder state, s_t is a decoder state
    # attn_vec_size is the length of the vectors v, b_attn, (W_h h_i) and (W_s s_t).
    # We set it to be equal to the size of the encoder states.
    attention_vec_size = attn_size

    # Get the weight matrix W_h and apply it to each encoder state to get (W_h h_i), the encoder features
    W_h = variable_scope.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
    question_features = nn_ops.conv2d(question_states, W_h, [1, 1, 1, 1], "SAME") # shape (batch_size,attn_length,1,attention_vec_size)
    W_r = variable_scope.get_variable("W_r", [1, 1, attn_size, attention_vec_size])
    review_features = nn_ops.conv2d(review_states, W_r, [1, 1, 1, 1], "SAME")  # shape (batch_size,review_num*attn_length,1,attention_vec_size)

    W_s = variable_scope.get_variable("W_o", [1, 1, s_attn_size, attention_vec_size])
    sent_features = nn_ops.conv2d(sent_states, W_s, [1, 1, 1, 1], "SAME")  # shape (batch_size,review_num*attn_length,1,attention_vec_size)


    # Get the weight vectors v and w_c (w_c is for coverage)
    v_q = variable_scope.get_variable("v_q", [attention_vec_size])
    v_r = variable_scope.get_variable("v_r", [attention_vec_size])
    v_s = variable_scope.get_variable("v_s", [attention_vec_size])
    if use_coverage:
      with variable_scope.variable_scope("coverage"):
        w_q_c = variable_scope.get_variable("w_q_c", [1, 1, 1, attention_vec_size])
        w_r_c = variable_scope.get_variable("w_r_c", [1, 1, 1, attention_vec_size])

    if q_prev_coverage is not None: # for beam search mode with coverage
      # reshape from (batch_size, attn_length) to (batch_size, attn_len, 1, 1)
      q_prev_coverage = tf.expand_dims(tf.expand_dims(q_prev_coverage,2),3)
    if r_prev_coverage is not None: # for beam search mode with coverage
      # reshape from (batch_size, attn_length) to (batch_size, attn_len, 1, 1)
      r_prev_coverage = tf.expand_dims(tf.expand_dims(r_prev_coverage,2),3)

    def s_attention(decoder_features):
      with variable_scope.variable_scope("Sentence_Attention"):
        e = math_ops.reduce_sum(v_s * math_ops.tanh(sent_features + decoder_features), [2, 3])  # shape (batch_size,attn_length)
        sent_attention = masked_attention_q(e, ds_padding_mask)
      return sent_attention
    
    def q_attention(decoder_state, coverage=None):
      """Calculate the context vector and attention distribution from the decoder state.

      Args:
        decoder_state: state of the decoder
        coverage: Optional. Previous timestep's coverage vector, shape (batch_size, attn_len, 1, 1).

      Returns:
        context_vector: weighted sum of encoder_states
        attn_dist: attention distribution
        coverage: new coverage vector. shape (batch_size, attn_len, 1, 1)
      """
      with variable_scope.variable_scope("Question_Attention"):
        # Pass the decoder state through a linear layer (this is W_s s_t + b_attn in the paper)
        decoder_features = linear(decoder_state, attention_vec_size, True) # shape (batch_size, attention_vec_size)
        decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1) # reshape to (batch_size, 1, 1, attention_vec_size)

        if use_coverage and coverage is not None: # non-first step of coverage
          # Multiply coverage vector by w_c to get coverage_features.
          coverage_features = nn_ops.conv2d(coverage, w_q_c, [1, 1, 1, 1], "SAME") # c has shape (batch_size, attn_length, 1, attention_vec_size)

          # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
          e = math_ops.reduce_sum(v_q * math_ops.tanh(question_features + decoder_features + coverage_features), [2, 3])  # shape (batch_size,attn_length)

          # Calculate attention distribution
          attn_dist = masked_attention_q(e, q_padding_mask)

          # Update coverage vector
          coverage += array_ops.reshape(attn_dist, [batch_size, -1, 1, 1])
        else:
          # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
          e = math_ops.reduce_sum(v_q * math_ops.tanh(question_features + decoder_features), [2, 3]) # calculate e

          # Calculate attention distribution
          attn_dist = masked_attention_q(e, q_padding_mask)

          if use_coverage: # first step of training
            coverage = tf.expand_dims(tf.expand_dims(attn_dist,2),2) # initialize coverage

        # Calculate the context vector from attn_dist and encoder_states
        context_vector = math_ops.reduce_sum(array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * question_states, [1, 2]) # shape (batch_size, attn_size).
        context_vector = array_ops.reshape(context_vector, [-1, attn_size])

      return context_vector, attn_dist, coverage

    def r_attention(decoder_state, sent_attention, coverage=None):
      """Calculate the context vector and attention distribution from the decoder state.

      Args:
        decoder_state: state of the decoder
        coverage: Optional. Previous timestep's coverage vector, shape (batch_size, attn_len, 1, 1).

      Returns:
        context_vector: weighted sum of encoder_states
        attn_dist: attention distribution
        coverage: new coverage vector. shape (batch_size, attn_len, 1, 1)
      """
      with variable_scope.variable_scope("Review_Attention"):
        # Pass the decoder state through a linear layer (this is W_s s_t + b_attn in the paper)
        decoder_features = linear(decoder_state, attention_vec_size, True) # shape (batch_size, attention_vec_size)
        decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1) # reshape to (batch_size, 1, 1, attention_vec_size)

        if use_coverage and coverage is not None: # non-first step of coverage
          # Multiply coverage vector by w_c to get coverage_features.
          coverage_features = nn_ops.conv2d(coverage, w_r_c, [1, 1, 1, 1], "SAME") # c has shape (batch_size, attn_length, 1, attention_vec_size)

          # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
          e = math_ops.reduce_sum(v_r * math_ops.tanh(review_features + decoder_features + coverage_features), [2, 3])  # shape (batch_size,attn_length)

          # Calculate attention distribution
          attn_dist = masked_attention_r(e, r_padding_mask, sent_attention, batch_size, num)

          # Update coverage vector
          coverage += array_ops.reshape(attn_dist, [batch_size, -1, 1, 1])
        else:
          # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
          e = math_ops.reduce_sum(v_r * math_ops.tanh(review_features + decoder_features), [2, 3]) # calculate e

          # Calculate attention distribution
          attn_dist = masked_attention_r(e, r_padding_mask, sent_attention, batch_size, num)

          if use_coverage: # first step of training
            coverage = tf.expand_dims(tf.expand_dims(attn_dist,2),2) # initialize coverage

        # Calculate the context vector from attn_dist and encoder_states
        context_vector = math_ops.reduce_sum(array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * review_states, [1, 2]) # shape (batch_size, attn_size).
        context_vector = array_ops.reshape(context_vector, [-1, attn_size])

      return context_vector, attn_dist, coverage

    outputs = []
    q_attn_dists = []
    r_attn_dists = []
    p_gens = []
    state = initial_state
    output_states = []
    q_coverage = q_prev_coverage # initialize coverage to None or whatever was passed in
    r_coverage = r_prev_coverage
    q_context_vector = array_ops.zeros([batch_size, attn_size])
    q_context_vector.set_shape([None, attn_size])  # Ensure the second shape of attention vectors is set.
    r_context_vector = array_ops.zeros([batch_size, attn_size])
    r_context_vector.set_shape([None, attn_size])  # Ensure the second shape of attention vectors is set.
    if initial_state_attention: # true in decode mode
      # Re-calculate the context vector from the previous step so that we can pass it through a linear layer with this step's input to get a modified version of the input
      with variable_scope.variable_scope('Decoder'):
        decoder_features = linear(initial_state, attention_vec_size, True) # shape (batch_size, attention_vec_size)
        decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1) # reshape to (batch_size, 1, 1, attention_vec_size)
      sent_attention = s_attention(decoder_features)
      q_context_vector, _, q_coverage = q_attention(initial_state, q_coverage) # in decode mode, this is what updates the coverage vector
      r_context_vector, _, r_coverage = r_attention(initial_state, sent_attention, r_coverage)
    for i, inp in enumerate(decoder_inputs):
      tf.logging.info("Adding attention_decoder timestep %i of %i", i, len(decoder_inputs))
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()

      # Merge input and previous attentions into one vector x of the same size as inp
      input_size = inp.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)
      x = linear([inp] + [q_context_vector] + [r_context_vector], input_size, True)
      #x = inp

      # Run the decoder RNN cell. cell_output = decoder state
      cell_output, state = cell(x, state)
      output_states.append(cell_output)

      # Run the attention mechanism.
      if i == 0 and initial_state_attention:  # always true in decode mode
        with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True): # you need this because you've already run the initial attention(...) call
          with variable_scope.variable_scope('Decoder'):
            decoder_features = linear(initial_state, attention_vec_size, True) # shape (batch_size, attention_vec_size)
            decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1) # reshape to (batch_size, 1, 1, attention_vec_size)
          sent_attention = s_attention(decoder_features)
          q_context_vector, q_attn_dist, _ = q_attention(state, q_coverage) # don't allow coverage to update
          r_context_vector, r_attn_dist, _ = r_attention(state, sent_attention, r_coverage)
      else:
        with variable_scope.variable_scope('Decoder'):
          decoder_features = linear(initial_state, attention_vec_size, True) # shape (batch_size, attention_vec_size)
          decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1) # reshape to (batch_size, 1, 1, attention_vec_size)
        sent_attention = s_attention(decoder_features)
        q_context_vector, q_attn_dist, q_coverage = q_attention(state, q_coverage)
        r_context_vector, r_attn_dist, r_coverage = r_attention(state, sent_attention, r_coverage)
      q_attn_dists.append(q_attn_dist)
      r_attn_dists.append(r_attn_dist)

      # Calculate p_gen
      if pointer_gen:
        with tf.variable_scope('calculate_pgen'):
          #p_gen = linear([q_context_vector, r_context_vector, state.c, state.h, x], 3, True) # a scalar
          p_gen = linear([cell_output] + [q_context_vector] + [r_context_vector], 3, True)  # a scalar
          p_gen = tf.nn.softmax(p_gen)
          p_gen = tf.split(p_gen, 3, 1)
          p_gens.append(p_gen)

      # Concatenate the cell_output (= decoder state) and the context vector, and pass them through a linear layer
      # This is V[s_t, h*_t] + b in the paper
      with variable_scope.variable_scope("AttnOutputProjection"):
        output = linear([cell_output] + [q_context_vector] + [r_context_vector], cell.output_size, True)
        #output = linear([cell_output], cell.output_size, True)
      outputs.append(output)

    # If using coverage, reshape it
    if r_coverage is not None:
      r_coverage = array_ops.reshape(r_coverage, [batch_size, -1])
    if q_coverage is not None:
      q_coverage = array_ops.reshape(q_coverage, [batch_size, -1])

    return outputs, output_states, state, q_attn_dists, r_attn_dists, p_gens, q_coverage, r_coverage


def masked_attention_q(e, padding_mask):
  """Take softmax of e then apply enc_padding_mask and re-normalize"""
  attn_dist = nn_ops.softmax(e)+1e-8 # take softmax. shape (batch_size, attn_length)
  attn_dist *= padding_mask # apply mask
  masked_sums = tf.reduce_sum(attn_dist, axis=1) # shape (batch_size)
  return attn_dist / tf.reshape(masked_sums, [-1, 1]) # re-normalize

'''
def masked_attention_r(e, padding_mask, review_attention, batch_size, num):
  attn_dist = nn_ops.softmax(e)+1e-8 # take softmax. shape (batch_size, attn_length)
  attn_dist *= padding_mask # apply mask
  masked_sums = tf.reduce_sum(attn_dist, axis=1) # shape (batch_size)
  attn_dist /= tf.reshape(masked_sums, [-1, 1]) # re-normalize
  
  review_attention = tf.expand_dims(review_attention, -1)
  attn_dist = tf.reshape(attn_dist, [batch_size, num, -1])
  attn_dist = tf.multiply(attn_dist, review_attention)
  attn_norm = tf.reshape(attn_dist, [batch_size, -1])
  combined_attn_dist = tf.divide(attn_norm, tf.expand_dims(tf.reduce_sum(attn_norm, 1), -1))
  return combined_attn_dist
'''

def masked_attention_r(e, padding_mask, review_attention, batch_size, num):
  attn_dist = nn_ops.softmax(e)#+1e-8 # take softmax. shape (batch_size, attn_length)

  review_attention = tf.expand_dims(review_attention, -1)
  attn_dist = tf.reshape(attn_dist, [batch_size, num, -1])
  attn_dist = tf.multiply(attn_dist, review_attention)
  attn_norm = tf.reshape(attn_dist, [batch_size, -1])
  attn_dist = tf.divide(attn_norm, tf.expand_dims(tf.reduce_sum(attn_norm, 1), -1))

  #padding_mask *= review_attention
  padding_mask = tf.reshape(padding_mask, [batch_size, -1])
  attn_dist *= padding_mask # apply mask
  masked_sums = tf.reduce_sum(attn_dist, axis=1) # shape (batch_size)
  attn_dist /= tf.reshape(masked_sums, [-1, 1]) # re-normalize
  return attn_dist

def linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  total_arg_sizes = []
  for arg in args:
    if arg is None or (isinstance(arg, (list, tuple)) and not arg):
      raise ValueError("`arg` must be specified")
    if not isinstance(arg, (list, tuple)):
      arg = [arg]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in arg]
    for shape in shapes:
      if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
      if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
      else:
        total_arg_size += shape[1]
    total_arg_sizes.append(total_arg_size)

  # Now the computation.
  with tf.variable_scope(scope or "Linear"):
    res = []
    for i in range(len(args)):
      matrix = tf.get_variable("Matrix"+str(i), [total_arg_sizes[i], output_size])
      arg = args[i]
      if not isinstance(arg, (list, tuple)):
        arg = [arg]
      if len(arg) == 1:
        res.append(tf.matmul(arg[0], matrix))
      else:
        res.append(tf.matmul(tf.concat(axis=1, values=arg), matrix))
    if not bias:
      return tf.add_n(res)
    bias_term = tf.get_variable(
        "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
  return tf.add_n(res) + bias_term
