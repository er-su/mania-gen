import os
import json
import librosa
import numpy as np
from termcolor import colored
from dataset_preprocess import process_dataset

import collections

# Loading all features
def load_features(beatmap_folder, filename):
    name = filename.split(".")[0]
    mel_spectrogram = np.load(os.path.join(beatmap_folder, f"{name}_mel_spectrogram.npy"))
    onset_activation = np.load(os.path.join(beatmap_folder, f"{name}_onsets.npy"))
    beat_activation = np.load(os.path.join(beatmap_folder, f"{name}_beat_frames.npy"))
    tempo = np.load(os.path.join(beatmap_folder, f"{name}_tempo.npy"))

    if not mel_spectrogram.shape[1] == onset_activation.shape[0] == beat_activation.shape[0] == tempo.shape[0]:
        raise Exception("Error mismatching sizes in folder:" + beatmap_folder)
    
    return mel_spectrogram, onset_activation, beat_activation, tempo

# Load and align notes
def load_notes(notes_file, num_keys = 4):
    with open(notes_file, "r") as f:
        notes_data = json.load(f)

    return notes_data

def prepare_input_output(beatmap_folder, difficulty=4, num_keys=4): 
    if not os.path.isdir(beatmap_folder):
        return (None, None, None, None, None)
    
    if not os.path.exists(os.path.join(beatmap_folder, f"notes_{difficulty}.json")):
        print(colored(f"No matching difficulty in {beatmap_folder}", 'yellow')) 
        return (None, None, None, None, None)
        
    for file in os.listdir(beatmap_folder):
        if file.endswith(".wav"):
            duration = librosa.get_duration(path=os.path.join(beatmap_folder, file))
            filename = file
            break
        else:
            print(colored(f"Error finding audio path for {beatmap_folder}", 'red'))
            raise Exception()


    mel_spectrogram, onset_activation, beat_activation, tempo = load_features(beatmap_folder, filename)
    time_steps = mel_spectrogram.shape[1]
    mel_spectrogram = np.transpose(mel_spectrogram)

    input_data = np.stack([onset_activation, beat_activation, tempo], axis=-1)
    size = input_data.shape[0]

    #FIX LINE FOR WITHOUT
    notes = load_notes(os.path.join(beatmap_folder, f"notes_{difficulty}.json"))

    # Convert the notes to final form 
    short_note_matrix = np.zeros(shape=(time_steps, num_keys))
    long_note_matrix = np.zeros(shape=(time_steps, num_keys + 1))


    for note in notes:
        time = note["time"] / 1000  # Convert to seconds
        key = note["key"]
        is_hold = note["hold"]
        end_time = note["end_time"] / 1000

        index = min(int((time / duration) * time_steps), time_steps - 1)
        short_note_matrix[index, key] = 1
        long_note_matrix[index, -1] = time

        if is_hold:
            long_note_matrix[index, key] = end_time - time

    return mel_spectrogram, input_data, short_note_matrix, long_note_matrix, size



def load_data(main_folder, difficulty=4, num_keys=4):
    mega_mel = []
    mega_beats = []
    mega_short = []
    mega_long = []
    max_size = -1
    process_dataset("../data/beatmaps", num_keys)
    for folder in os.listdir(main_folder):
        mel, beats, short_labels, long_labels, size = prepare_input_output(os.path.join(main_folder, folder), difficulty=difficulty)
        if beats is not None and short_labels is not None:
            mega_mel.append(mel)
            mega_beats.append(beats)
            mega_short.append(short_labels)
            mega_long.append(long_labels)
            max_size = max_size if max_size > size else size

    if max_size == -1:
        raise Exception("Size not properly calculated")
    return mega_mel, mega_beats, mega_short, mega_long, max_size


if __name__ == "__main__":
    print(prepare_input_output("../data/processed_beatmaps/lmao"))
