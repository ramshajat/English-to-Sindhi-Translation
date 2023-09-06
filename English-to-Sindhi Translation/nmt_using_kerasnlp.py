from google.colab import drive
drive.mount('/content/drive')

import keras_nlp
import numpy as np
import pandas as pd
import pathlib
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import  model_from_json
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

BATCH_SIZE = 128
EPOCHS = 5  # This should be at least 10 for convergence
MAX_SEQUENCE_LENGTH = 20
ENG_VOCAB_SIZE = 104890
SND_VOCAB_SIZE = 104890

EMBED_DIM = 256
INTERMEDIATE_DIM = 2048
NUM_HEADS = 8

data = pd.read_csv('/content/drive/MyDrive/English-to-Sindhi Translation/english-sindhi2.csv')

text_pairs = data.values.tolist()

random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]

print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")

def train_word_piece(text_samples, vocab_size, reserved_tokens):
    bert_vocab_args = dict(
        # The target vocabulary size
        vocab_size=vocab_size,
        # Reserved tokens that must be included in the vocabulary
        reserved_tokens=reserved_tokens,
        # Arguments for `text.BertTokenizer`
        bert_tokenizer_params={"lower_case": True},
    )

    word_piece_ds = tf.data.Dataset.from_tensor_slices(text_samples)
    vocab = bert_vocab.bert_vocab_from_dataset(
        word_piece_ds.batch(1000).prefetch(2), **bert_vocab_args
    )
    return vocab

reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

eng_samples = [text_pair[0] for text_pair in train_pairs]
eng_vocab = train_word_piece(eng_samples, ENG_VOCAB_SIZE, reserved_tokens)

snd_samples = [text_pair[1] for text_pair in train_pairs]
snd_vocab = train_word_piece(snd_samples, SND_VOCAB_SIZE, reserved_tokens)

print("English Tokens: ", eng_vocab[100:110])
print("Sindhi Tokens: ", snd_vocab[100:110])

eng_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=eng_vocab, lowercase=False
)
snd_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=snd_vocab, lowercase=False
)

eng_input_ex = text_pairs[0][0]
eng_tokens_ex = eng_tokenizer.tokenize(eng_input_ex)
print("English sentence: ", eng_input_ex)
print("Tokens: ", eng_tokens_ex)
print("Recovered text after detokenizing: ", eng_tokenizer.detokenize(eng_tokens_ex))

print()

snd_input_ex = text_pairs[0][1]
snd_tokens_ex = snd_tokenizer.tokenize(snd_input_ex)
print("Sindhi sentence: ", snd_input_ex)
print("Tokens: ", snd_tokens_ex)
print("Recovered text after detokenizing: ", snd_tokenizer.detokenize(snd_tokens_ex))

def preprocess_batch(eng, snd):
    batch_size = tf.shape(snd)[0]

    eng = eng_tokenizer(eng)
    snd = snd_tokenizer(snd)

    # Pad `eng` to `MAX_SEQUENCE_LENGTH`.
    eng_start_end_packer = keras_nlp.layers.StartEndPacker(
        sequence_length=MAX_SEQUENCE_LENGTH,
        pad_value=eng_tokenizer.token_to_id("[PAD]"),
    )
    eng = eng_start_end_packer(eng)

    # Add special tokens (`"[START]"` and `"[END]"`) to `snd` and pad it as well.
    snd_start_end_packer = keras_nlp.layers.StartEndPacker(
        sequence_length=MAX_SEQUENCE_LENGTH + 1,
        start_value=snd_tokenizer.token_to_id("[START]"),
        end_value=snd_tokenizer.token_to_id("[END]"),
        pad_value=snd_tokenizer.token_to_id("[PAD]"),
    )
    snd = snd_start_end_packer(snd)

    return (
        {
            "encoder_inputs": eng,
            "decoder_inputs": snd[:, :-1],
        },
        snd[:, 1:],
    )


def make_dataset(pairs):
    eng_texts, snd_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    snd_texts = list(snd_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, snd_texts))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.shuffle(2048).prefetch(16).cache()


train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

for inputs, targets in train_ds.take(1):
    print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
    print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
    print(f"targets.shape: {targets.shape}")

# Encoder
encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")

x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=ENG_VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)(encoder_inputs)

encoder_outputs = keras_nlp.layers.TransformerEncoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(inputs=x)
encoder = keras.Model(encoder_inputs, encoder_outputs)


# Decoder
decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
encoded_seq_inputs = keras.Input(shape=(None, EMBED_DIM), name="decoder_state_inputs")

x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=SND_VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)(decoder_inputs)

x = keras_nlp.layers.TransformerDecoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(decoder_sequence=x, encoder_sequence=encoded_seq_inputs)
x = keras.layers.Dropout(0.5)(x)
decoder_outputs = keras.layers.Dense(SND_VOCAB_SIZE, activation="softmax")(x)
decoder = keras.Model(
    [
        decoder_inputs,
        encoded_seq_inputs,
    ],
    decoder_outputs,
)
decoder_outputs = decoder([decoder_inputs, encoder_outputs])

transformer = keras.Model(
    [encoder_inputs, decoder_inputs],
    decoder_outputs,
    name="transformer",
)

transformer.summary()
transformer.compile(
    "rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
hist = transformer.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)