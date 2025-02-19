# TODO: Replace if there is dup folder during creation
# Make so that if there is more than 1 difficulty it'll be fine

import os
import json
import math
from extract_preprocess import extract_features
from extract_preprocess import process_audio
from termcolor import colored, cprint


PROCESSED_DIR = "../data/processed_beatmaps"
PREPROCESSED_DIR = "../data/beatmaps"

def parse_osu_file(osu_file):
    with open(osu_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    hit_objects_start = False
    beatmap_data = []

    num_keys = -1
    difficulty = -1
    audio_filename = ""

    for line in lines:
        line = line.strip()

        # Check audio filename
        if "AudioFilename: " in line:
            audio_filename = line.split(":")[-1].strip()

        # Ensure it is mania gamemode
        if "Mode:" in line:
            if "3" not in line:
                print(colored(f"{osu_file} is not mania gamemode", 'yellow'))
                return None
        
        # Check keysize
        if "CircleSize:" in line:
            num_keys = int(line[-1])

        # Find difficulty
        if "Version:" in line:
            lower = line.lower()

            if "easy" in lower:
                difficulty = 0
            elif "normal" in lower:
                difficulty = 1
            elif "hard" in lower:
                difficulty = 2
            elif "insane" in lower:
                difficulty = 3
            else:
                difficulty = 4
            continue
           
        if line == "[HitObjects]":
            hit_objects_start = True
            continue

        if hit_objects_start and line:
            parts = line.split(",")
            x_position = int(parts[0])
            timestamp = int(parts[2])
            object_type = int(parts[3])
            end_timestamp = None

            key = math.floor(64 * num_keys / 512)

            is_hold = bool(object_type & 128)
            if is_hold:
                end_timestamp = int(parts[5].split(":")[0])

            beatmap_data.append({
                "time": timestamp,
                "key": key,
                "hold": is_hold,
                "end_time": end_timestamp if is_hold else timestamp
            })

    # print(beatmap_data)
    # print(difficulty)
    if num_keys == -1 or difficulty == -1:
        raise Exception(f"Issue parsing beatmap {osu_file}. Either number of keys was not parsed or difficulty was not parsed")
    
    return beatmap_data, difficulty, num_keys, audio_filename

def process_dataset(dataset_dir, num_keys=4):
    count = 0
    for beatmap_folder in os.listdir(dataset_dir):
        pre_folder_path = os.path.join(dataset_dir, beatmap_folder)
        if not os.path.isdir(pre_folder_path):
            continue
        
        # Create new dir inside training folder and number it
        new_folder_path = os.path.join(PROCESSED_DIR, beatmap_folder)
        try:
            os.mkdir(path=new_folder_path)
        except FileExistsError:
            # FIX THIS LATERRR
            # IF IT ALR EXISTS, PROBABLY CAN SKIP
            # MAYBE ADD SCRIPT TO CHECK IF ITS VALID
            #
            print(colored(f"The beatmap {beatmap_folder} has already been processed, please reverify", 'yellow'))
            continue
            #os.rmdir(new_folder_path)
            #os.mkdir(path=new_folder_path)

        
        #print(new_folder_path)
        #print(pre_folder_path)

        # Now process audio and extract features
        
        audio_filename = ""
        # Go thru all files and create mega json
        for file in os.listdir(pre_folder_path):
            if file.endswith(".osu"):
                osu_path = os.path.join(pre_folder_path, file)
                osu_data, difficulty, keys, audio_filename = parse_osu_file(osu_path)

                if osu_data is None or keys != num_keys:
                    print(colored(f"{osu_path} is wrong num of keys", 'yellow'))
                    continue
                
                with open(os.path.join(new_folder_path, f"notes_{difficulty}.json"), "w") as f:
                    json.dump(osu_data, f, indent=2)
        
        audio_path = os.path.join(pre_folder_path, audio_filename)
        if not os.path.exists(audio_path):
            print(colored(f"Missing audio file in folder {beatmap_folder}", 'red'))
            continue

        processed_wav_path = process_audio(pre_folder_path, new_folder_path, audio_filename)
        # print(f"Found {audio_filename} as audio file")
        
        extract_features(new_folder_path, processed_wav_path, os.path.split(processed_wav_path)[-1])

        count += 1
        print(colored(f"Finished processing {new_folder_path}", 'green'))

    print(f"Finished processing {count} beatmaps")


if __name__ == "__main__":
    process_dataset("../data/beatmaps")


        