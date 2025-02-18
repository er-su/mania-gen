import os
import librosa 
import numpy as np
import soundfile as sf
from pydub import AudioSegment



# Converts mp3 to wav
def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(file=mp3_path)
    audio.export(wav_path, format="wav")
    print(f"Converted {mp3_path} to {wav_path}")

def convert_to_wav(non_path, wav_path):
    audio = AudioSegment.from_file(file=non_path, format="ogg")
    audio.export(wav_path, format="wav")
    print(f"Converted {non_path} to {wav_path}")

# Normalize and returns waveform + samplerate
def load_and_normalize_audio(wav_path, target_sr=22050):
    y, sr =librosa.load(wav_path, sr=target_sr)
    y = librosa.util.normalize(y)
    return y, sr

def process_audio(dir_path, write_path, file_name):
    wav_path = file_name
     
    if file_name.endswith(".wav"):
        wav_path = os.path.join(write_path, file_name)

    elif file_name.endswith(".mp3"):
        mp3_path = os.path.join(dir_path, file_name)
        wav_path = os.path.join(write_path, file_name.replace(".mp3", ".wav"))
        convert_mp3_to_wav(mp3_path, wav_path)

    elif file_name.endswith(".ogg"):
        ogg_path = os.path.join(dir_path, file_name)
        wav_path = os.path.join(write_path, file_name.replace(".ogg", ".wav"))
        convert_to_wav(ogg_path, wav_path)

    else:
        raise Exception("Unsupported audio type")

    y, sr = load_and_normalize_audio(wav_path)

    processed_wav_path = wav_path.replace(".wav", "_processed.wav")
    sf.write(processed_wav_path, y, sr)

    print(f"Processed and saved: {processed_wav_path}")

    os.remove(wav_path)

    return processed_wav_path

# Extracts mel spectrogram from audio signal
def extract_mel_spectrogram(y, sr, hop_length=441):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

# Beat detection of audio signal
def extract_onsets(y, sr, mel_spectrogram, hop_length=441): 
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    onset_activation = np.zeros(mel_spectrogram.shape[1])
    onset_activation[onset_frames] = 1

    for x in onset_frames:
        if onset_activation[x] != 1:
            raise Exception("Error with onset activation array")

    return onset_activation

def extract_tempo_and_beats(y, sr, mel_spectrogram):
    # Extract onset envelope for tempo and beat detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    # Detect tempo (beats per minute) and beat frames
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    beat_activation = np.zeros(mel_spectrogram.shape[1])
    beat_activation[beat_frames] = 1

    tempo_feature = np.full((mel_spectrogram.shape[1],), round(tempo[0]))

    for x in beat_frames:
        if beat_activation[x] != 1:
            raise Exception("Error with onset beat array")
    
    return tempo_feature, beat_activation

def extract_features(write_path, processed_wav_path, file_name):

    # Loading in processed data
    #processed_wav_path = os.path.join(dir_path, file_name)
    y, sr = librosa.load(processed_wav_path, sr=22050)

    # Extracting relevant features
    mel_spectrogram = extract_mel_spectrogram(y, sr)
    onsets = extract_onsets(y, sr, mel_spectrogram)
    tempo, beat_frames = extract_tempo_and_beats(y, sr, mel_spectrogram)

    # Saving related data paths
    mel_spectrogram_path = os.path.join(write_path, file_name.replace(".wav", "_mel_spectrogram.npy"))
    onsets_path = os.path.join(write_path, file_name.replace(".wav", "_onsets.npy"))
    tempo_path = os.path.join(write_path, file_name.replace(".wav", "_tempo.npy"))
    beat_frames_path = os.path.join(write_path, file_name.replace(".wav", "_beat_frames.npy"))

    # Saving data at paths
    np.save(mel_spectrogram_path, mel_spectrogram)
    np.save(onsets_path, onsets)
    np.save(tempo_path, tempo)
    np.save(beat_frames_path, beat_frames)

    print(f"Features for {write_path} extracted and saved.")


""" # CHAGNE LATER
if __name__ == "__main__":
    files = os.listdir(RAW_DATA_DIR)
    mp3_files = [file for file in files if file.endswith(".mp3")]

    if not mp3_files:
        print("No MP3 files found in the raw data directory")
    else:
        for file in mp3_files:
            process_audio(file) """

