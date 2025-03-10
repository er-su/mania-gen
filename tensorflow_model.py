# DISCLAIMER! DISCLAIMER! DISCLAIMER! DISCLAIMER! DISCLAIMER!
# INITIAL ATTEMPT USING KERAS LIBRARY. KEPT FOR EDUCATIONAL
# AND REFERENCE PURPOSES. DOES NOT FUNCTION AS INTENDED. DO NOT RUN

import keras
import numpy as np
from keras import layers
from keras import activations
import tensorflow as tf

hyperparams = {
      "batch_size": 16,
      "num_epochs": 1,
      "action_emb_dim": 32,
      "beat_frac_embd_dim": 32,
      "beat_num_embd_dim": 8,
      "onset_gru_output_size": 128,
      "action_gru_output_size": 128,
      "onset_as_input_size": 128,
      "onset_dense_size": 128,
      "action_dense_size": 128,
      "learning_rate": 0.001
}

class OsuGen(keras.Model):
        
    def __init__(self, hyperparams, num_keys=4):
        super(OsuGen, self).__init__()
        self.bf_embd_dim = hyperparams["beat_frac_embd_dim"]
        self.bn_embd_dim = hyperparams["beat_num_embd_dim"]
        self.onset_gru_size = hyperparams["onset_gru_output_size"]
        self.action_gru_size = hyperparams["action_gru_output_size"]
        self.onset_as_input = hyperparams["onset_as_input_size"]
        self.onset_dense_size = hyperparams["onset_dense_size"]
        self.action_dense_size = hyperparams["action_dense_size"]
        self.num_combos = 4 ** num_keys
        self.batch_size = hyperparams["batch_size"]

        self.convo = keras.Sequential([
              #layers.Input(shape=(3, None, 80), name="mel"),
              layers.Conv2D(8, (5,3), strides=(1,2), padding="same", data_format="channels_first"),
              layers.BatchNormalization(axis=1),
              layers.ReLU(),
              layers.Conv2D(16, (5,3), strides=(1,2), padding="same", data_format="channels_first"),
              layers.BatchNormalization(axis=1),
              layers.ReLU(),
              layers.Conv2D(32, (5,3), strides=(1,2), padding="same", data_format="channels_first"),
              layers.BatchNormalization(axis=1),
              layers.ReLU(),
              layers.Conv2D(64, (5,3), strides=(1,2), padding="same", data_format="channels_first"),
              layers.BatchNormalization(axis=1),
              layers.ReLU()], name="mel convolution")
        
        self.permute = layers.Permute((2, 1, 3))
        self.reshape = layers.Reshape((-1, 3 * 80))
        self.concat = layers.Concatenate(axis=-1)

        # Generating onsets
        self.onset_GRU = layers.GRU(self.onset_gru_size, return_sequences=True)
        self.onset_dense = keras.Sequential([
              layers.TimeDistributed(layers.Dense(self.onset_dense_size, activation="gelu")),
              layers.TimeDistributed(layers.Dense(1, activation="sigmoid", name="onsets"))
              ])
                
        # Generating actions
        self.onset_as_input = layers.TimeDistributed(layers.Dense(self.onset_as_input))
        self.action_GRU = keras.Sequential([
              layers.GRU(self.action_gru_size, return_sequences=True),
              layers.TimeDistributed(layers.Dense(self.action_dense_size, activation="gelu")),
              layers.TimeDistributed(layers.Dense(self.num_combos, activation="sigmoid", name="actions"))
        ])
        self.beat_frac_embd = layers.Embedding(49, self.bf_embd_dim)
        self.beat_num_embd = layers.Embedding(4, self.bn_embd_dim)

        self.squeeze = keras.layers.Lambda(lambda x: keras.backend.squeeze(x, axis=2), output_shape=(None, None))
    
    def call(self, inputs):
        mel, beat_frac, beat_num = inputs
        # mel -> tensor of shape (batch, 3, num_timesteps, 80)
        # beat_frac -> tensor of shape (num_timesteps, 1)
        # beat_num -> tensor of shape (num_timesteps, 1)
        
        # Get onset outputs
        conv_output = self.convo(mel)
        conv_output = self.permute(conv_output)
        conv_output = self.reshape(conv_output)

        bf_embd = self.squeeze(self.beat_frac_embd(beat_frac))
        bn_embd = self.squeeze(self.beat_num_embd(beat_num))
        
        onset_in = self.concat([conv_output, bf_embd, bn_embd])

        onset_gru_out = self.onset_GRU(onset_in)
        onset_out = self.onset_dense(onset_gru_out)

        # Get action outputs
        action_in = self.concat([onset_in, onset_out])
        action_out = self.action_GRU(action_in)

        return onset_out, action_out
    
    def build_graph(self):
        mel = layers.Input(shape=(3, 1000, 80), name="mel")
        beat_fracs = layers.Input(shape=(1000, 1), name="beat fracs")
        beat_num = layers.Input(shape=(1000, 1), name="beat nums")
        return keras.Model(inputs=[mel, beat_fracs, beat_num], outputs=self.call([mel, beat_fracs, beat_num]))

        
class SqueezeLayer(keras.Layer):
    def call(self, input_tensor):
        return tf.squeeze(input_tensor)
