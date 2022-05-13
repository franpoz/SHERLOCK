import time

import keras
import numpy as np
from keras import layers
import tensorflow as tf


class SANTO:
    """
    Self-Attention Neural Network for Transiting Objects
    """
    def __init__(self) -> None:
        super().__init__()

    def loss_function(self, loss_holder, target, pred):
        mask = tf.math.logical_not(tf.math.equal(target, 0))
        loss_ = loss_holder(target, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def main_train(self, dataset, n_epochs, print_every=50):
        tf.keras.backend.clear_session()
        input_dim = 15000
        output_dim = input_dim
        checkpoint_path = "/mnt/DATA-2/training_data/SANTEX/checkpoint"
        # Create the Transformer model
        transformer = Transformer(num_heads=4, key_dim=input_dim, output_dim=output_dim, stack_depth=2, dropout_rate=0.1)

        # Define a categorical cross entropy loss
        loss_holder = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
        # Define a metric to store the mean loss of every epoch
        train_loss = tf.keras.metrics.Mean(name="train_loss")
        # Define a metric to save the accuracy in every epoch
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
        # Create the scheduler for learning rate decay
        leaning_rate = CustomSchedule(input_dim)
        # Create the Adam optimizer (Vaswani et al., 2017)
        optimizer = tf.keras.optimizers.Adam(leaning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        # Create the Checkpoint
        ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Last checkpoint restored.")

        ''' Train the transformer model for n_epochs using the data generator dataset'''
        losses = []
        accuracies = []
        # In every epoch
        for epoch in range(n_epochs):
            print("Start of epoch {}".format(epoch + 1))
            start = time.time()
            # Reset the loss and accuracy calculations
            train_loss.reset_states()
            train_accuracy.reset_states()
            # Get a batch of inputs and targets
            for (batch, (enc_inputs, targets)) in enumerate(dataset):
                # Set the decoder inputs
                dec_inputs = targets[:, :-1]
                # Set the target outputs, right shifted
                dec_outputs_real = targets[:, 1:]
                with tf.GradientTape() as tape:
                    # Call the transformer and get the predicted output
                    predictions = transformer(enc_inputs, dec_inputs, True)
                    # Calculate the loss
                    loss = self.loss_function(dec_outputs_real, predictions, loss_holder)
                # Update the weights and optimizer
                gradients = tape.gradient(loss, transformer.trainable_variables)
                optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
                # Save and store the metrics
                train_loss(loss)
                train_accuracy(dec_outputs_real, predictions)

                if batch % print_every == 0:
                    losses.append(train_loss.result())
                    accuracies.append(train_accuracy.result())
                    print("Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}".format(
                        epoch + 1, batch, train_loss.result(), train_accuracy.result()))
            # Checkpoint the model on every epoch
            ckpt_save_path = ckpt_manager.save()
            print("Saving checkpoint for epoch {} in {}".format(epoch + 1, ckpt_save_path))
            print("Time for 1 epoch: {} secs\n".format(time.time() - start))
        return losses, accuracies


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, max_points_in_sequences, embed_dim=1):
        super(TokenAndPositionEmbedding, self).__init__()
        # TODO this positional encoding should be improved because it is restricted to the max sequence length from
        #  the training data
        self.pos_emb = layers.Embedding(input_dim=max_points_in_sequences, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


class TransformerEncoder(layers.Layer):
    def __init__(self, key_dim, num_heads, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.att = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(self.input_dim, activation="relu"), layers.Dense(self.key_dim), ]
        )
        self.att_norm = layers.LayerNormalization(epsilon=1e-6)
        self.ffn_norm = layers.LayerNormalization(epsilon=1e-6)
        self.att_dropout = layers.Dropout(self.dropout_rate)
        self.ffn_dropout = layers.Dropout(self.dropout_rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.att_dropout(attn_output, training=training)
        out1 = self.att_norm(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.ffn_dropout(ffn_output, training=training)
        return self.ffn_norm(out1 + ffn_output)


class TransformerDecoder(layers.Layer):
    def __init__(self, key_dim, num_heads, dropout_rate=0.1):
        super(TransformerDecoder, self).__init__()
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.att1 = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
        self.att1_dropout = layers.Dropout(self.dropout_rate)
        self.att1_norm = layers.LayerNormalization(epsilon=1e-6)
        self.att2 = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
        self.att2_dropout = layers.Dropout(self.dropout_rate)
        self.att2_norm = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = keras.Sequential(
            [layers.Dense(self.input_shape, activation="relu"), layers.Dense(self.key_dim), ]
        )
        self.ffn_dropout = layers.Dropout(self.dropout_rate)
        self.ffn_norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, encoder_inputs, output_inputs, attention_mask, training):
        attn_output = self.att1(output_inputs, output_inputs, attention_mask=attention_mask)
        attn_output = self.att1_dropout(attn_output, training=training)
        norm1_output = self.att1_norm(output_inputs + attn_output)
        attn_output = self.att2(norm1_output, encoder_inputs)
        attn_output = self.att2_dropout(attn_output, training=training)
        norm2_output = self.att1_norm(norm1_output + attn_output)
        ffn_output = self.ffn(norm2_output)
        ffn_output = self.ffn_dropout(ffn_output, training=training)
        return self.ffn_norm(norm2_output + ffn_output)


class Transformer(layers.Layer):
    def __init__(self, key_dim, num_heads, output_dim, stack_depth=2, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.key_dim = key_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.stack_depth = stack_depth

    def build(self, input_shape):
        self.input_dim = input_shape
        self.encoders = np.array([])
        for i in np.arange(0, self.stack_depth):
            self.encoders.append(TransformerEncoder(self.key_dim, self.num_heads, self.dropout_rate))
        self.decoders = np.array([])
        for i in np.arange(0, self.stack_depth):
            self.decoders.append(TransformerEncoder(self.key_dim, self.num_heads, self.dropout_rate))
        self.average_pooling = layers.GlobalAveragePooling1D()
        self.average_pooling_dropout = layers.Dropout(self.dropout_rate)
        self.pooling_dense = layers.Dense(self.output_dim, activation="relu")
        self.pooling_dense_dropout = layers.Dropout(self.dropout_rate)
        self.output_dense = layers.Dense(self.output_dim, activation="softmax")

    def call(self, input, predicted_output, attention_mask, training):
        for encoder in self.encoders:
            encoder_output = encoder(input, training)
        for decoder in self.decoders:
            decoder_output = decoder(encoder_output, predicted_output, attention_mask, training)
        linear_proj_output = self.average_pooling(decoder_output)
        linear_proj_output = self.pooling_dense(linear_proj_output)
        linear_proj_output = self.pooling_dense_dropout(linear_proj_output, training=training)
        linear_proj_output = self.output_dense(linear_proj_output, training=training)
        return linear_proj_output


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
