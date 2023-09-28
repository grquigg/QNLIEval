import os

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_datasets as tfds

tfhub_handle_encoder = "https://tfhub.dev/tensorflow/albert_en_xxlarge/2"

class Classifier(tf.keras.Model):
    def __init__(self, num_classes):
      super(Classifier, self).__init__(name="prediction")
      self.encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True)
      self.dropout = tf.keras.layers.Dropout(0.1)
      self.dense = tf.keras.layers.Dense(num_classes)

    def call(self, preprocessed_text):
      encoder_outputs = self.encoder(preprocessed_text)
      pooled_output = encoder_outputs["pooled_output"]
      x = self.dropout(pooled_output)
      x = self.dense(x)
      return x

model = Classifier(2)
init_lr = 1e-5
warmup_steps = 1986
checkpoint_path ="best_model" 
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

metrics = tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)

model.compile(loss=loss, metrics=[metrics])
model.load_weights(checkpoint_path)
