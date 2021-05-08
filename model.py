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

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import time
import numpy as np
import tensorflow as tf
from attention_decoder import attention_decoder
from tensorflow.contrib.tensorboard.plugins import projector
import data

FLAGS = tf.app.flags.FLAGS

class SummarizationModel(object):
  """A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode, and coverage"""

  def __init__(self, hps, vocab):
    self._hps = hps
    self._vocab = vocab

  def _add_placeholders(self):
    """Add placeholders to the graph. These are entry points for any input data."""
    hps = self._hps

    # encoder part
    self._d_batch = tf.compat.v1.placeholder(tf.int32, [hps.batch_size, hps.sent_num, None], name='d_batch')
    self._d_lens = tf.compat.v1.placeholder(tf.int32, [hps.batch_size, hps.sent_num], name='d_lens')
    self._ds_lens = tf.compat.v1.placeholder(tf.int32, [hps.batch_size], name='ds_lens')
    self._q_batch = tf.compat.v1.placeholder(tf.int32, [hps.batch_size, None], name='q_batch')
    self._q_lens = tf.compat.v1.placeholder(tf.int32, [hps.batch_size], name='q_lens')
    self._d_padding_mask = tf.compat.v1.placeholder(tf.float32, [hps.batch_size,  hps.sent_num, None], name='d_padding_mask')
    self._ds_padding_mask = tf.compat.v1.placeholder(tf.float32, [hps.batch_size,  hps.sent_num], name='ds_padding_mask')
    self._q_padding_mask = tf.compat.v1.placeholder(tf.float32, [hps.batch_size, None], name='q_padding_mask')
    if FLAGS.pointer_gen:
      self._d_batch_extend_vocab = tf.compat.v1.placeholder(tf.int32, [hps.batch_size, hps.sent_num, None], name='d_batch_extend_vocab')
      self._q_batch_extend_vocab = tf.compat.v1.placeholder(tf.int32, [hps.batch_size, None], name='q_batch_extend_vocab')
      self._max_oovs = tf.compat.v1.placeholder(tf.int32, [], name='max_oovs')

    # decoder part
    self._dec_batch = tf.compat.v1.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
    self._target_batch = tf.compat.v1.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
    self._dec_padding_mask = tf.compat.v1.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps], name='dec_padding_mask')

    if hps.mode=="decode" and hps.coverage:
      self.q_prev_coverage = tf.compat.v1.placeholder(tf.float32, [hps.batch_size, None], name='q_prev_coverage')
      self.d_prev_coverage = tf.compat.v1.placeholder(tf.float32, [hps.batch_size, None], name='d_prev_coverage')


  def _make_feed_dict(self, batch, just_enc=False):
    """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

    Args:
      batch: Batch object
      just_enc: Boolean. If True, only feed the parts needed for the encoder.
    """
    feed_dict = {}
    feed_dict[self._d_batch] = batch.d_batch
    feed_dict[self._d_lens] = batch.d_lens
    feed_dict[self._q_batch] = batch.q_batch
    feed_dict[self._q_lens] = batch.q_lens
    feed_dict[self._ds_lens] = batch.ds_lens
    feed_dict[self._d_padding_mask] = batch.d_padding_mask
    feed_dict[self._ds_padding_mask] = batch.ds_padding_mask
    feed_dict[self._q_padding_mask] = batch.q_padding_mask
    if FLAGS.pointer_gen:
      feed_dict[self._d_batch_extend_vocab] = batch.d_batch_extend_vocab
      feed_dict[self._q_batch_extend_vocab] = batch.q_batch_extend_vocab
      feed_dict[self._max_oovs] = batch.max_oovs
    if not just_enc:
      feed_dict[self._dec_batch] = batch.dec_batch
      feed_dict[self._target_batch] = batch.target_batch
      feed_dict[self._dec_padding_mask] = batch.dec_padding_mask
    return feed_dict

  def _add_encoder(self, encoder_inputs, seq_len, scope, dropout=0.8):
    """Add a single-layer bidirectional LSTM encoder to the graph.

    Args:
      encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
      seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

    Returns:
      encoder_outputs:
        A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
      fw_state, bw_state:
        Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    """
    with tf.compat.v1.variable_scope('encoder' + scope):
      cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=dropout)
      cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=dropout)
      (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
      encoder_outputs = tf.concat(axis=2, values=encoder_outputs) # concatenate the forwards and backwards states
    return encoder_outputs, fw_st, bw_st


  def _reduce_states(self, fw_st, bw_st, d_fw_st, d_bw_st):
    """Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.

    Args:
      fw_st: LSTMStateTuple with hidden_dim units.
      bw_st: LSTMStateTuple with hidden_dim units.

    Returns:
      state: LSTMStateTuple with hidden_dim units.
    """
    hidden_dim = self._hps.hidden_dim
    #memory_dim = memory.get_shape()[1]
    with tf.compat.v1.variable_scope('reduce_final_st'):

      # Define weights and biases to reduce the cell and reduce the state
      w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 4, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 4, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

      # Apply linear layer
      old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c, d_fw_st.c, d_bw_st.c]) # Concatenation of fw and bw cell
      old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h, d_fw_st.h, d_bw_st.h]) # Concatenation of fw and bw state
      new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c) # Get new cell from old cell
      new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h) # Get new state from old state
      return tf.contrib.rnn.LSTMStateTuple(new_c, new_h) # Return new cell and state


  def _add_decoder(self, inputs):
    """Add attention decoder to the graph. In train or eval mode, you call this once to get output on ALL steps. In decode (beam search) mode, you call this once for EACH decoder step.

    Args:
      inputs: inputs to the decoder (word embeddings). A list of tensors shape (batch_size, emb_dim)

    Returns:
      outputs: List of tensors; the outputs of the decoder
      out_state: The final state of the decoder
      attn_dists: A list of tensors; the attention distributions
      p_gens: A list of scalar tensors; the generation probabilities
      coverage: A tensor, the current coverage vector
    """
    hps = self._hps
    cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)

    q_prev_coverage = self.q_prev_coverage if hps.mode=="decode" and hps.coverage else None # In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time
    d_prev_coverage = self.d_prev_coverage if hps.mode=="decode" and hps.coverage else None # In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time

    outputs, output_states, out_state, q_attn_dists, d_attn_dists, p_gens, q_coverage, d_coverage = attention_decoder(inputs, self._dec_in_state, self.dec_question_inputs, self.dec_document_inputs, self.dec_sent_inputs, hps.sent_num, self._q_padding_mask, self._d_padding_mask, self._ds_padding_mask, cell, initial_state_attention=(hps.mode=="decode"), pointer_gen=hps.pointer_gen, use_coverage=hps.coverage, q_prev_coverage=q_prev_coverage, r_prev_coverage=d_prev_coverage)

    return outputs, output_states, out_state, q_attn_dists, d_attn_dists, p_gens, q_coverage, d_coverage

  def _calc_final_dist(self, vocab_dists, q_attn_dists, d_attn_dists):
    """Calculate the final distribution, for the pointer-generator model

    Args:
      vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
      attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays

    Returns:
      final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
    """
    with tf.compat.v1.variable_scope('final_distribution'):
      # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
      vocab_dists = [p_gen[0] * dist for (p_gen,dist) in zip(self.p_gens, vocab_dists)]
      q_attn_dists = [p_gen[1] * dist for (p_gen,dist) in zip(self.p_gens, q_attn_dists)]
      d_attn_dists = [p_gen[2] * dist for (p_gen, dist) in zip(self.p_gens, d_attn_dists)]

      # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
      extended_vsize = self._vocab.size() + self._max_oovs # the maximum (over the batch) size of the extended vocabulary
      extra_zeros = tf.zeros((self._hps.batch_size, self._max_oovs))
      vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists] # list length max_dec_steps of shape (batch_size, extended_vsize)

      # Project the values in the attention distributions onto the appropriate entries in the final distributions
      # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
      # This is done for each decoder timestep.
      # This is fiddly; we use tf.scatter_nd to do the projection
      d_batch_extend_vocab = tf.layers.flatten(self._d_batch_extend_vocab)
      batch_nums = tf.range(0, limit=self._hps.batch_size) # shape (batch_size)
      batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
      d_attn_len = tf.shape(self._d_batch_extend_vocab)[1] * tf.shape(self._d_batch_extend_vocab)[2] # number of states we attend over
      q_attn_len = tf.shape(self._q_batch_extend_vocab)[1]
      d_batch_nums = tf.tile(batch_nums, [1, d_attn_len]) # shape (batch_size, attn_len)
      q_batch_nums = tf.tile(batch_nums, [1, q_attn_len])
      d_indices = tf.stack((d_batch_nums, d_batch_extend_vocab), axis=2) # shape (batch_size, enc_t, 2)
      q_indices = tf.stack((q_batch_nums, self._q_batch_extend_vocab), axis=2)
      shape = [self._hps.batch_size, extended_vsize]
      d_attn_dists_projected = [tf.scatter_nd(d_indices, copy_dist, shape) for copy_dist in d_attn_dists] # list length max_dec_steps (batch_size, extended_vsize)
      q_attn_dists_projected = [tf.scatter_nd(q_indices, copy_dist, shape) for copy_dist in q_attn_dists]

      # Add the vocab distributions and the copy distributions together to get the final distributions
      # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
      # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
      final_dists = [vocab_dist + d_copy_dist + q_copy_dist for (vocab_dist,d_copy_dist,q_copy_dist) in zip(vocab_dists_extended, d_attn_dists_projected, q_attn_dists_projected)]

      return final_dists

  def _add_emb_vis(self, embedding_var):
    """Do setup so that we can view word embedding visualization in Tensorboard, as described here:
    https://www.tensorflow.org/get_started/embedding_viz
    Make the vocab metadata file, then make the projector config file pointing to it."""
    train_dir = os.path.join(FLAGS.log_root, "train")
    vocab_metadata_path = os.path.join(train_dir, "vocab_metadata.tsv")
    self._vocab.write_metadata(vocab_metadata_path) # write metadata file
    summary_writer = tf.summary.FileWriter(train_dir)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = vocab_metadata_path
    projector.visualize_embeddings(summary_writer, config)


  def _word_compare_aggregate(self, question, reviews, batch_size, num):
    dim1 = int(question.get_shape()[2])
    dim2 = int(reviews.get_shape()[2])
    U = tf.get_variable(
      "U",
      shape=[dim1, dim2],
      initializer=tf.contrib.layers.xavier_initializer())

    question = tf.expand_dims(question, 1)
    question = tf.tile(question, (1, num, 1, 1))
    q_padding_mask = tf.expand_dims(self._q_padding_mask, 1)
    q_padding_mask = tf.tile(q_padding_mask, (1, num, 1))

    question = tf.reshape(question, [batch_size*num, -1, dim1])
    transform_left = tf.einsum('ijk,kl->ijl', question, U)
    att_mat = tf.matmul(transform_left, tf.transpose(reviews, [0,2,1]))

    r_attn = tf.nn.softmax(tf.reduce_max(att_mat, axis=1)) # (batch_size*review_num, r_len)
    r_attn *= tf.reshape(self._d_padding_mask, [batch_size*num, -1])
    masked_sums = tf.reduce_sum(r_attn, axis=1)+1e-8  # (batch_size*review_num)
    r_attn = r_attn / tf.reshape(masked_sums, [-1, 1])  # re-normalize
    r_attn = tf.expand_dims(r_attn, -1, name='review_attention')

    q_attn = tf.nn.softmax(tf.reduce_max(att_mat, axis=2))  # (batch_size*review_num, q_len)
    q_attn *= tf.reshape(q_padding_mask, [batch_size * num, -1])
    masked_sums = tf.reduce_sum(q_attn, axis=1)+1e-8  # (batch_size*review_num)
    q_attn = q_attn / tf.reshape(masked_sums, [-1, 1])  # re-normalize
    q_attn = tf.expand_dims(q_attn, -1, name='question_attention')

    dec_review_inputs = tf.multiply(reviews, r_attn) # (batch_size * review_num, r_len, dim)
    dec_question_inputs = tf.multiply(question, q_attn) # (batch_size * review_num, q_len, dim)
    att_review_outputs = tf.reshape(tf.reduce_sum(dec_review_inputs,1),[-1,dim2])
    att_question_outputs = tf.reshape(tf.reduce_sum(dec_question_inputs,1),[-1,dim1])

    dec_question_inputs = tf.reshape(dec_question_inputs, [batch_size, num, -1, dim1])
    dec_question_inputs = tf.reduce_mean(dec_question_inputs, 1) # (batch_size, q_len, dim)
    dec_review_inputs = tf.reshape(dec_review_inputs, [batch_size, num, -1, dim2]) # (batch_size, review_num, r_len, dim)
    #dec_review_inputs = tf.reshape(dec_review_inputs, [batch_size, -1, dim2]) # (batch_size, review_num * r_len, dim)
    
    return dec_question_inputs, dec_review_inputs


  def _sent_compare_aggregate(self, doc_sent_inputs, q_inputs, ds_lens, padding_mask, hops, hidden_size=256, dropout_keep_prob=0.8):
    sent_inputs, ds_fw_st, ds_bw_st = self._add_encoder(doc_sent_inputs, ds_lens, "/sentence") # batch_size, ds_len, hidden_dim*2
        
    q_inputs = tf.reduce_sum(q_inputs, axis=1) # batch_size, dim
    batch_size = int(sent_inputs.get_shape()[0])
    num = int(sent_inputs.get_shape()[1])
    dim1 = int(sent_inputs.get_shape()[2])
    dim2 = int(q_inputs.get_shape()[1])

    W = tf.get_variable(
          "W", shape=[dim1, dim2],
          initializer=tf.contrib.layers.xavier_initializer())
    transform_left = tf.einsum('ijk,kl->ijl', sent_inputs, W)
    sim = tf.matmul(transform_left, tf.expand_dims(q_inputs, -1)) 
    sim = tf.reshape(sim, [batch_size, num])

    sim = tf.nn.softmax(sim)
    sim *= padding_mask
    masked_sums = tf.reduce_sum(sim, axis=1)
    sent_attention = sim / tf.reshape(masked_sums, [-1, 1])  # re-normalize

    sent_attention = tf.expand_dims(sent_attention, -1)
    att_sent_inputs = tf.multiply(sent_inputs, sent_attention)
    dec_in_state = tf.reshape(tf.reduce_sum(att_sent_inputs, 1),[batch_size, -1])

    if hops == 1:
      return sent_attention, att_sent_inputs, dec_in_state, ds_fw_st, ds_bw_st
    else:
      for hop in range(hops-1):
        sent_outputs, ds_fw_st, ds_bw_st = self._add_encoder(att_sent_inputs, ds_lens, "/sentence_hop%s"%hop) # batch_size, ds_len, hidden_dim*2
        '''
        W = tf.get_variable(
          "W_hop%s"%hop, shape=[dim1, dim1],
          initializer=tf.contrib.layers.xavier_initializer())
        transform_left = tf.einsum('ijk,kl->ijl', sent_outputs, W)
        sim = tf.matmul(transform_left, tf.expand_dims(last_hop, -1)) 
        sim = tf.reshape(sim, [batch_size, num])

        sim = tf.nn.softmax(sim)
        sim *= padding_mask
        masked_sums = tf.reduce_sum(sim, axis=1)
        sent_attention = sim / tf.reshape(masked_sums, [-1, 1])  # re-normalize
        '''
        U = tf.get_variable(
          "U_hop%s"%hop, shape=[dim1, dim1],
          initializer=tf.contrib.layers.xavier_initializer())
        transform_left = tf.einsum('ijk,kl->ijl', sent_outputs, U)
        att_mat = tf.matmul(transform_left, tf.transpose(sent_inputs, [0,2,1])) # batch_size, sent_num, sent_num

        sim_padding_mask = tf.expand_dims(padding_mask, 1)
        sim_padding_mask = tf.tile(sim_padding_mask, (1, num, 1))
        tmp_mask = tf.ones([batch_size, num, num]) - tf.matrix_diag(tf.ones([batch_size, num]))
        sim_padding_mask *= tmp_mask
        sim_padding_mask *= tf.expand_dims(padding_mask, -1)
        att_mat *= sim_padding_mask
        attn = tf.nn.softmax(tf.tanh(att_mat)) + 1e-8 # batch_size, sent_num, sent_num
        masked_sums = tf.reduce_sum(attn, axis=1)  
        attn = attn / tf.reshape(masked_sums, [batch_size, num, 1])  # re-normalize
        scores = tf.reduce_max(attn, axis=1)

        W = tf.get_variable(
              "W_hop%s"%hop, shape=[dim1, dim2],
              initializer=tf.contrib.layers.xavier_initializer())
        transform_left = tf.einsum('ijk,kl->ijl', sent_inputs, W)
        sim = tf.matmul(transform_left, tf.expand_dims(q_inputs, -1)) 
        sim = tf.reshape(sim, [batch_size, num])

        mmr = 0.5 * sim + 0.5 * scores
        mmr *= padding_mask
        mmr = tf.nn.softmax(mmr)
        mmr *= padding_mask
        masked_sums = tf.reduce_sum(mmr, axis=1)
        sent_attention = mmr / tf.reshape(masked_sums, [-1, 1])  # re-normalize

        sent_attention = tf.expand_dims(sent_attention, -1)
        att_sent_inputs = tf.multiply(sent_outputs, sent_attention)
        dec_in_state = tf.reshape(tf.reduce_sum(att_sent_inputs, 1),[batch_size, -1])
      return sent_attention, att_sent_inputs, dec_in_state, ds_fw_st, ds_bw_st
  


  def _add_seq2seq(self):
    """Add the whole sequence-to-sequence model to the graph."""
    hps = self._hps
    vsize = self._vocab.size() # size of the vocabulary
    vocab = self._vocab

    with tf.compat.v1.variable_scope('seq2seq'):
      # Some initializers
      self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
      self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)

      # Add embedding matrix (shared by the encoder and decoder inputs)
      with tf.compat.v1.variable_scope('embedding'):
        if hps.glove:
          init_embeddings = data.get_init_embeddings(vocab._id_to_word, hps.emb_dim)
          embedding = tf.get_variable("embedding", initializer=init_embeddings)
        else:
          embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
        if hps.mode=="train": self._add_emb_vis(embedding) # add to tensorboard
        emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._d_batch) # tensor with shape (batch_size, review_num, max_enc_steps, emb_size)
        emb_dec_inputs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(self._dec_batch, axis=1)] # list length max_dec_steps containing shape (batch_size, emb_size)
        emb_q_inputs = tf.nn.embedding_lookup(embedding, self._q_batch) # tensor with shape (batch_size, max_enc_steps, emb_size)

      # Add the encoder.
      d_lens = tf.reshape(self._d_lens, [-1])
      emb_enc_inputs = tf.reshape(emb_enc_inputs, [hps.batch_size * hps.sent_num, -1, hps.emb_dim])
      enc_outputs, _, _ = self._add_encoder(emb_enc_inputs, d_lens, "/document") # (batch_size*review_num, enc_lens, emb_size)
      q_outputs, q_fw_st, q_bw_st = self._add_encoder(emb_q_inputs, self._q_lens, "/question") # (batch_size, q_lens, emb_size)

      self.dec_question_inputs = q_outputs
      self.dec_document_inputs = tf.reshape(enc_outputs, [hps.batch_size, -1 , int(enc_outputs.get_shape()[-1])])

      # word_compare_aggregate layer
      with tf.compat.v1.variable_scope('word_compare_aggregate'):
        dec_question_inputs, dec_document_inputs = self._word_compare_aggregate(q_outputs, enc_outputs, hps.batch_size, hps.sent_num)

      # sentence_compare_aggregate layer
      with tf.compat.v1.variable_scope('sent_compare_aggregate'):
        doc_sent_inputs = tf.reduce_sum(dec_document_inputs, axis=2)
        _, att_sent_inputs, decode_in_state, ds_fw_st, ds_bw_st = self._sent_compare_aggregate(doc_sent_inputs, dec_question_inputs, self._ds_lens, self._ds_padding_mask, hps.hops)

      self.dec_sent_inputs = att_sent_inputs

      # Our encoder is bidirectional and our decoder is unidirectional so we need to reduce the final encoder hidden state to the right size to be the initial decoder hidden state
      self._dec_in_state = self._reduce_states(q_fw_st, q_bw_st, ds_fw_st, ds_bw_st)

      # Add the decoder.
      with tf.compat.v1.variable_scope('decoder'):
        decoder_outputs, output_states, self._dec_out_state, self.q_attn_dists, self.d_attn_dists, self.p_gens, self.q_coverage, self.d_coverage = self._add_decoder(emb_dec_inputs)

      if hps.mode in ['train', 'decode']:
        # Add the output projection to obtain the vocabulary distribution
        with tf.compat.v1.variable_scope('output_projection'):
          w = tf.get_variable('w', [hps.hidden_dim, vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
          v = tf.get_variable('v', [vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
          vocab_scores = [] # vocab_scores is the vocabulary distribution before applying softmax. Each entry on the list corresponds to one decoder step
          for i,output in enumerate(decoder_outputs):
            if i > 0:
              tf.get_variable_scope().reuse_variables()
            vocab_scores.append(tf.nn.xw_plus_b(output, w, v)) # apply the linear layer

          vocab_dists = [tf.nn.softmax(s) for s in vocab_scores] # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.


        # For pointer-generator model, calc final distribution from copy distribution and vocabulary distribution
        if FLAGS.pointer_gen:
          final_dists = self._calc_final_dist(vocab_dists,  self.q_attn_dists, self.d_attn_dists)
        else: # final distribution is just vocabulary distribution
          final_dists = vocab_dists


      if hps.mode in ['train']:
        # Calculate the loss
        with tf.compat.v1.variable_scope('loss'):
          if FLAGS.pointer_gen:
            # Calculate the loss per step
            # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
            loss_per_step = [] # will be list length max_dec_steps containing shape (batch_size)
            batch_nums = tf.range(0, limit=hps.batch_size) # shape (batch_size)
            for dec_step, dist in enumerate(final_dists):
              targets = self._target_batch[:,dec_step] # The indices of the target words. shape (batch_size)
              indices = tf.stack((batch_nums, targets), axis=1) # shape (batch_size, 2)
              gold_probs = tf.gather_nd(dist, indices) # shape (batch_size). prob of correct words on this step
              #losses = -tf.log(gold_probs)
              losses = -tf.log(tf.clip_by_value(gold_probs, 1e-8, tf.reduce_max(gold_probs)))
              loss_per_step.append(losses)

            # Apply dec_padding_mask and get loss
            self._loss_sum = _mask_and_avg(loss_per_step, self._dec_padding_mask)

          else: # baseline model
            self._loss_sum = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1), self._target_batch, self._dec_padding_mask) # this applies softmax internally

          self._loss = self._loss_sum
          tf.summary.scalar('loss', self._loss)

          # Calculate coverage loss from the attention distributions
          if hps.coverage:
            with tf.variable_scope('coverage_loss'):
              self._q_coverage_loss, self._d_coverage_loss = _coverage_loss(self.q_attn_dists, self.d_attn_dists, self.p_gens, self._dec_padding_mask)
              self._coverage_loss = self._q_coverage_loss + self._d_coverage_loss
              tf.summary.scalar('coverage_loss', self._coverage_loss)
            self._total_loss = self._loss + hps.cov_loss_wt * self._coverage_loss
            tf.summary.scalar('total_loss', self._total_loss)

    if hps.mode == "decode":
      # We run decode beam search mode one decoder step at a time
      assert len(final_dists)==1 # final_dists is a singleton list containing shape (batch_size, extended_vsize)
      final_dists = final_dists[0]
      topk_probs, self._topk_ids = tf.nn.top_k(final_dists, hps.batch_size*2) # take the k largest probs. note batch_size=beam_size in decode mode
      self._topk_log_probs = tf.log(topk_probs)


  def _add_train_op(self):
    """Sets self._train_op, the op to run for training."""
    # Take gradients of the trainable variables w.r.t. the loss function to minimize
    loss_to_minimize = self._total_loss if self._hps.coverage else self._loss
    tvars = tf.trainable_variables()
    gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

    # Clip the gradients
    with tf.device("/gpu:1"):
      grads, _ = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)

    # Add a summary
    #tf.summary.scalar('global_norm', global_norm)

    # Apply adagrad optimizer
    optimizer = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)
    with tf.device("/gpu:1"):
      self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')


  def build_graph(self):
    """Add the placeholders, model, global step, train_op and summaries to the graph"""
    tf.compat.v1.logging.info('Building graph...')
    t0 = time.time()
    self._add_placeholders()
    with tf.device("/gpu:0"):
      self._add_seq2seq()
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if self._hps.mode == 'train':
      self._add_train_op()
    self._summaries = tf.summary.merge_all()
    t1 = time.time()
    tf.compat.v1.logging.info('Time to build graph: %i seconds', t1 - t0)

  def run_train_step(self, sess, batch):
    """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'train_op': self._train_op,
        'summaries': self._summaries,
        'loss': self._loss,
        'global_step': self.global_step,
    }
    if self._hps.coverage:
      to_return['q_coverage'] = self._q_coverage_loss,
      to_return['d_coverage'] = self._d_coverage_loss,
      to_return['coverage_loss'] = self._coverage_loss
    return sess.run(to_return, feed_dict)

  def run_eval_step(self, sess, batch):
    """Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'loss': self._loss,
    }
    if self._hps.coverage:
      to_return['total_loss'] = self._total_loss
    return sess.run(to_return, feed_dict)

  def run_encoder(self, sess, batch):
    """For beam search decoding. Run the encoder on the batch and return the encoder states and decoder initial state.

    Args:
      sess: Tensorflow session.
      batch: Batch object that is the same example repeated across the batch (for beam search)

    Returns:
      enc_states: The encoder states. A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim].
      dec_in_state: A LSTMStateTuple of shape ([1,hidden_dim],[1,hidden_dim])
    """
    feed_dict = self._make_feed_dict(batch, just_enc=True) # feed the batch into the placeholders
    (dec_in_state, dec_question_inputs, dec_document_inputs, global_step) = \
    sess.run([self._dec_in_state, self.dec_question_inputs, self.dec_document_inputs, self.global_step], feed_dict) # run the encoder

    # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
    dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
    return dec_in_state, dec_question_inputs, dec_document_inputs


  def decode_onestep(self, sess, batch, latest_tokens, dec_init_states, dec_question_inputs, dec_document_inputs, q_prev_coverage, r_prev_coverage):
    """For beam search decoding. Run the decoder for one step.

    Args:
      sess: Tensorflow session.
      batch: Batch object containing single example repeated across the batch
      latest_tokens: Tokens to be fed as input into the decoder for this timestep
      enc_states: The encoder states.
      dec_init_states: List of beam_size LSTMStateTuples; the decoder states from the previous timestep
      prev_coverage: List of np arrays. The coverage vectors from the previous timestep. List of None if not using coverage.

    Returns:
      ids: top 2k ids. shape [beam_size, 2*beam_size]
      probs: top 2k log probabilities. shape [beam_size, 2*beam_size]
      new_states: new states of the decoder. a list length beam_size containing
        LSTMStateTuples each of shape ([hidden_dim,],[hidden_dim,])
      attn_dists: List length beam_size containing lists length attn_length.
      p_gens: Generation probabilities for this step. A list length beam_size. List of None if in baseline mode.
      new_coverage: Coverage vectors for this step. A list of arrays. List of None if coverage is not turned on.
    """

    beam_size = len(dec_init_states)

    # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
    cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
    hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
    new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
    new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
    new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

    feed = {
        self.dec_question_inputs: dec_question_inputs,
        self.dec_document_inputs: dec_document_inputs,
        self._d_padding_mask: batch.d_padding_mask,
        self._q_padding_mask: batch.q_padding_mask,
        self._ds_padding_mask: batch.ds_padding_mask,
        self._q_lens: batch.q_lens,
        self._q_batch: batch.q_batch,
        self._d_batch: batch.d_batch,
        self._d_lens: batch.d_lens,
        self._ds_lens: batch.ds_lens,
        self._dec_in_state: new_dec_in_state,
        self._dec_batch: np.transpose(np.array([latest_tokens])),
    }

    to_return = {
      "ids": self._topk_ids,
      "probs": self._topk_log_probs,
      "states": self._dec_out_state,
      "q_attn_dists": self.q_attn_dists,
      "d_attn_dists": self.d_attn_dists,
    }

    if FLAGS.pointer_gen:
      feed[self._q_batch_extend_vocab] = batch.q_batch_extend_vocab
      feed[self._d_batch_extend_vocab] = batch.d_batch_extend_vocab
      feed[self._max_oovs] = batch.max_oovs
      to_return['p_gens'] = self.p_gens

    if self._hps.coverage:
      feed[self.q_prev_coverage] = np.stack(q_prev_coverage, axis=0)
      feed[self.d_prev_coverage] = np.stack(r_prev_coverage, axis=0)
      to_return['q_coverage'] = self.q_coverage
      to_return['d_coverage'] = self.d_coverage

    results = sess.run(to_return, feed_dict=feed) # run the decoder step

    # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
    new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in range(beam_size)]

    # Convert singleton list containing a tensor to a list of k arrays
    assert len(results['q_attn_dists'])==1
    q_attn_dists = results['q_attn_dists'][0].tolist()
    assert len(results['d_attn_dists']) == 1
    d_attn_dists = results['d_attn_dists'][0].tolist()

    if FLAGS.pointer_gen:
      # Convert singleton list containing a tensor to a list of k arrays
      assert len(results['p_gens'])==1
      p_gens = np.transpose(np.asarray(results['p_gens'][0]))[0].tolist()
    else:
      p_gens = [None for _ in range(beam_size)]

    # Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
    if FLAGS.coverage:
      q_new_coverage = results['q_coverage'].tolist()
      r_new_coverage = results['d_coverage'].tolist()
      assert len(q_new_coverage) == beam_size
      assert len(r_new_coverage) == beam_size
    else:
      q_new_coverage = [None for _ in range(beam_size)]
      r_new_coverage = [None for _ in range(beam_size)]

    return results['ids'], results['probs'], new_states, q_attn_dists, d_attn_dists, p_gens, q_new_coverage, r_new_coverage


def _mask_and_avg(values, padding_mask):
  """Applies mask to values then returns overall average (a scalar)

  Args:
    values: a list length max_dec_steps containing arrays shape (batch_size).
    padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

  Returns:
    a scalar
  """

  dec_lens = tf.reduce_sum(padding_mask, axis=1) # shape batch_size. float32
  values_per_step = [v * padding_mask[:,dec_step] for dec_step,v in enumerate(values)]
  values_per_ex = sum(values_per_step)/dec_lens # shape (batch_size); normalized value for each batch member
  return tf.reduce_mean(values_per_ex) # overall average


def _coverage_loss(q_attn_dists, r_attn_dists, p_gens, padding_mask):
  """Calculates the coverage loss from the attention distributions.

  Args:
    attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
    padding_mask: shape (batch_size, max_dec_steps).

  Returns:
    coverage_loss: scalar
  """
  q_attn_dists = [p_gen[1]/(p_gen[1]+p_gen[2]) * dist for (p_gen, dist) in zip(p_gens, q_attn_dists)]
  r_attn_dists = [p_gen[2]/(p_gen[1]+p_gen[2]) * dist for (p_gen, dist) in zip(p_gens, r_attn_dists)]
  q_coverage = tf.zeros_like(q_attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
  q_covlosses = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
  for a in q_attn_dists:
    covloss = tf.reduce_sum(tf.minimum(a, q_coverage), [1]) # calculate the coverage loss for this step
    q_covlosses.append(covloss)
    q_coverage += a # update the coverage vector
  q_coverage_loss = _mask_and_avg(q_covlosses, padding_mask)
  r_coverage = tf.zeros_like(r_attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
  r_covlosses = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
  for a in r_attn_dists:
    covloss = tf.reduce_sum(tf.minimum(a, r_coverage), [1]) # calculate the coverage loss for this step
    r_covlosses.append(covloss)
    r_coverage += a # update the coverage vector
  r_coverage_loss = _mask_and_avg(r_covlosses, padding_mask)

  return q_coverage_loss, r_coverage_loss


