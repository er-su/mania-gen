import keras
import json
import numpy as np
from keras import layers
from keras import ops
from data_loader import load_data

from data_loader import prepare_input_output

def build_model(input_shape, num_keys=4):
    model = keras.models.Sequential()

    # Assm model
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Masking(0.0))

    model.add(layers.GRU(128, return_sequences=True))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    model.add(layers.GRU(128, return_sequences=True))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    output_units = num_keys * 2 + 1
    model.add(layers.TimeDistributed(layers.Dense(output_units, activation="sigmoid")))

    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(), metrics=["accuracy"])

    return model

def functional_build_model(input_shape, mel_shape, num_keys=4):
    data_input = layers.Input(shape=input_shape, name="beats", sparse=True)
    mel_input = layers.Input(shape=mel_shape, name="mel")

    # Gru 1 
    mel_features = layers.GRU(128, return_sequences=True)(mel_input)
    #mel_features = layers.BatchNormalization()(x)
    #xmel_features = layers.Dropout(0.2)(x)

    data_features = layers.GRU(128, return_sequences=True)(data_input)
    #data_features = layers.BatchNormalization()(x)
    #data_features = layers.Dropout(0.2)(x)

    x = layers.concatenate([data_features, mel_features])

    # Gru 2
    x = layers.GRU(128, return_sequences=True)(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.Dropout(0.2)(x)

    # Time Dist 1 for Short
    norm_notes = layers.TimeDistributed(layers.Dense(num_keys, activation="sigmoid", name="short"))(x)
    # Time Dist 2 for Long
    held_notes = layers.TimeDistributed(layers.Dense(num_keys + 1, activation="relu", name="long"))(x)

    model = keras.Model(
        inputs=[data_input, mel_input],
        outputs={"short": norm_notes, "long": held_notes},
    )

    model.compile(
        optimizer=keras.optimizers.Adam(), 
        loss={
            "short": keras.losses.BinaryCrossentropy(),
            "long": keras.losses.MeanAbsoluteError(),
        },
        loss_weights={"short": 1.0, "long": 0.5},
    )

    model.summary()

    return model

def train_model(model, mel, beats, short, long, epochs=1, batch_size=1):
    for t, step_mel in enumerate(mel):
        mel2 = np.array([step_mel])
        beats2 = np.array([beats[t]])
        short2 = np.array([short[t]])
        long2 = np.array([long[t]])
        model.fit(
            {"mel": mel2, "beats": beats2},
            {"short": short2, "long": long2},
            epochs=epochs, 
            batch_size=batch_size,
        )

def save_model(model, filepath="default_name.keras"):
    model.save(filepath)

def load_model(filepath="default_name.keras"):
    return keras.models.load_model(filepath)

def inference(model, input_data, input_mel):
    input_data = np.asarray([input_data])
    input_mel = np.asarray([input_mel])
    predictions = model({"beats": input_data, "mel": input_mel}, training=False)

    return predictions

if __name__ == "__main__":
    #mel, beats, short, long, size = prepare_input_output("../data/processed_beatmaps/lmao", difficulty=3)
    mega_mel, mega_beats, mega_short, mega_long, max_size = load_data("../data/processed_beatmaps", difficulty=4)
    model = functional_build_model(input_shape=(None, 3), mel_shape=(None, 128))
    #train_model(model, mega_mel, mega_beats, mega_short, mega_long)
    save_model(model)
    #print(inference(model, beats, mel))