import shutil
import math
import json
import librosa
import numpy as np
from pathlib import Path
from termcolor import colored

DIFFICULTY_0 = [" ez","beginner","easy","light"]
DIFFICULTY_1 = [" nm","standard","normal"]
DIFFICULTY_2 = ["hard","advanced"," hd"]
DIFFICULTY_3 = ["insane"," in","another"]
DIFFICULTY_4 = ["expert","ex","hyper","master"]

DIFFICULTIES = [DIFFICULTY_0,
                DIFFICULTY_1,
                DIFFICULTY_2,
                DIFFICULTY_3,
                DIFFICULTY_4]

FILETYPES = [".mp3", ".wav", ".ogg"]

class Converter:
    def __init__(self,
                 beatmap_list_path: Path,
                 output_folder_path: Path,
                 n_fft_list=[1024, 2048, 4096],
                 hop=10, #in ms
                 beat_frac_size=48):
        
        self.beatmap_list_path = beatmap_list_path
        self.output_folder_path = output_folder_path
        self.hop = hop
        self.hop_len = int((hop / 1000) * 44100)
        self.n_fft_list = n_fft_list
        self.beat_frac_size = beat_frac_size

    def beat_frac(self, time, offset, bpm):
        beat_frac = ((time - offset) % bpm) / bpm
        beat_frac = round(beat_frac * self.beat_frac_size)
        return beat_frac
    
    def beat_num(self, time, offset, bpm, meter=4):
        return math.floor(((time - offset) / bpm) % meter)

    def convert_audio(self, y, offset, bpm):
        mel = []
        for n_fft_val in self.n_fft_list:
            mel.append(librosa.feature.melspectrogram(y=y,
                                                       sr=44100,
                                                       n_fft=n_fft_val,
                                                       hop_length=self.hop_len,
                                                       n_mels=80,
                                                       power=2))
        mel = np.array(mel)
        mel = np.transpose(mel, (0, 2, 1))

        #Check
        beat_frac = [self.beat_frac(x * self.hop, offset, bpm) for x in range(mel.shape[1])]
        beat_num = [self.beat_num(x * self.hop, offset, bpm) for x in range(mel.shape[1])]

        return mel, beat_frac, beat_num
    
    def parse_osu(self, beatmap_folder, osu_fn):
        print(f"Parsing {osu_fn}")
        osu_path = beatmap_folder / osu_fn

        with open(osu_path, mode='r', encoding="utf-8") as f:
            data = f.read().splitlines()


        # Ensure it's mania
        i = data.index("[General]") + 1
        while data[i] != "" and "[" not in data[i]:
            #print(i)
            line = data[i].lower()
            if "AudioFilename:" in data[i]:
                audiofile = line.split(' ')[-1]
            
            if "mode:" in line:
                if int(line.split(' ')[-1]) != 3:
                    print(f"Beatmap {osu_fn} is not mania")
                    return [-1], -1, -1, -1, -1, -1
                else:
                    break
            i += 1

        # Get difficulty
        difficulty = -1
        version = -1
        i = data.index("[Metadata]") + 1
        while data[i] != "" and "[" not in data[i]:
            line = data[i].lower()
            if "version:" in line:
                version = i
                break
            i += 1
        
        #Check
        for dif_num, difficulties in enumerate(DIFFICULTIES):
            if any(dif_name in data[version].lower() for dif_name in difficulties):
                difficulty = dif_num
                break
            else:
                difficulty = 5
        
        # Get num keys
        num_keys = -1
        i = data.index("[Difficulty]") + 1
        while data[i] != "" and "[" not in data[i]:
            line = data[i].lower()
            if "circlesize:" in line:
                num_keys = int(line.split(':')[-1])
                break
            i += 1

        # Get BPM and offset
        # TODO: fix if same bpm but different offset midway thru
        timing_pts = []
        i = data.index("[TimingPoints]") + 1
        while data[i] != '' and "[" not in data[i]:
            timing_pts.append(data[i])
            i += 1


        bpms = []
        meters = []
        for tp in timing_pts:
            if float(tp.split(',')[1]) > 0: 
                bpms.append(float(tp.split(',')[1]))
            if int(tp.split(',')[2]) > 0:
                meters.append(int(tp.split(',')[2]))

        if bpms.count(bpms[0]) != len(bpms) or meters.count(meters[0]) != len(meters):
            print(colored(f"More than one meter or BPM in {osu_fn}", "yellow"))
            return [-1], -1, -1, -1, -1, -1
        if meters[0] != 4:
            print(colored(f"Multiple meters in {osu_fn}", "yellow"))
            return [-1], -1, -1, -1, -1, -1
        
        offset = float(timing_pts[0].split(',')[0])
        bpm = bpms[0]

        # Create obj list of all hit objects within the map
        obj_list = []
        objs = data[data.index("[HitObjects]") + 1:]
        for obj in objs:
            split = obj.split(',')
            key = math.floor(int(split[0]) * num_keys / 512)
            time = int(split[2])

            if int(split[3]) == 128:
                end_time = int(split[5].split(':')[0])
                obj_list.append([time, key, 2])
                obj_list.append([end_time, key, 3])
            else:
                obj_list.append([time, key, 1])
    
        obj_list = np.array(obj_list)
        obj_list[obj_list[:,0].argsort()]

        return obj_list, num_keys, offset, bpm, difficulty, audiofile
        
    def base_n_encoding(self, action_obj, num_keys):
        sum = 0
        for i, type in enumerate(action_obj):
            sum += type * (num_keys ** i)
        
        return sum

    def generate_labels(self, obj_list, num_timesteps, num_keys):

        # Obj_list is of size num_timesteps x 3 where each obj is [time, key, type 0-3]
        new_obj = obj_list.copy()
        for obj in new_obj:
            obj[0] = round(obj[0] / self.hop) # Value now represents index every 10ms

        # Convert to action_array of length num_timesteps, where the value is num_key-base encoded
        # for the combination of hits
        action_array = np.zeros((num_timesteps, num_keys))
        for obj in new_obj:
            time_in_10s, key, note_type = obj
            action_array[int(time_in_10s), int(key)] = note_type

        action_array = np.array([int(self.base_n_encoding(action_obj, num_keys)) for action_obj in action_array])
        
        # Then convert action array to one-hot encoding
        one_hot_array = np.zeros((action_array.size, 4 ** num_keys))
        one_hot_array[np.arange(action_array.size), action_array] = 1

        # Create onset array. 0 if nothing occurs, 1 if something occurs
        onset_array = [(1 if action_val > 0 else 0) for action_val in action_array]

        return one_hot_array, action_array, onset_array

    def verify(self, beatmap_folder: Path, num_parsed_osu):
        osu_files = list(beatmap_folder.rglob("*.npy"))

        if len(osu_files) != num_parsed_osu + 1:
            return False
        
        return (beatmap_folder / "audio_features.npy").exists()
        

    def convert(self, print_json=False):
        beatmap_list = self.beatmap_list_path.iterdir()
        beatmaps_info = []
        for beatmap_folder in beatmap_list:
            print(f"Processing {beatmap_folder.name}")
            for file in beatmap_folder.iterdir():

                if file.suffix == ".osu":
                    beatmap, num_keys, offset, bpm, difficulty, audiofile = self.parse_osu(beatmap_folder, file.name)
                    if num_keys != -1 or offset != -1:
                        beatmaps_info.append(
                            {"beatmap": beatmap,
                             "num_keys": num_keys,
                             "offset": offset,
                             "bpm": bpm,
                             "difficulty": difficulty,
                             "audiofile": audiofile,
                            }
                        )
            
            # Check if any valid maps
            if len(beatmaps_info) == 0:
                print(colored(f"No valid beatmaps found in {beatmap_folder.name}. Skipping beatmap", "red"))
                continue

            try:
                print(f"Loading audio {audiofile}")
                y, sr = librosa.load(beatmap_folder / audiofile, sr=44100, mono=True)

            except BaseException as err:
                print(colored(f"Unable to load audio in {beatmap_folder.name}. Skipping beatmap", "red"))
                continue
            
            mel, beat_frac, beat_num = self.convert_audio(y, offset, bpm)
            
            # Create and save into folder
            output_beatmap_folder: Path = self.output_folder_path / beatmap_folder.name

            try:
                output_beatmap_folder.mkdir(exist_ok=False)
            
            except FileExistsError as err:
                print(f"Folder {output_beatmap_folder.name} already exists. Verifying...")
                if not self.verify(output_beatmap_folder, len(beatmaps_info)):
                    print(colored(f"Issue with preexisiting folder. Reproecessing", "red"))
                    shutil.rmtree(output_beatmap_folder)
                    output_beatmap_folder.mkdir()
                else:
                    print(colored(f"No issues found. Skipping"))
                    continue

            np.save(output_beatmap_folder / "audio_features.npy", 
                    {"mel": mel,
                     "beat_frac": beat_frac,
                     "beat_num": beat_num})
            
            # Create set of all possible key types and save beatmaps in corresponding dirs
            num_keys_set = set()
            for beatmap in beatmaps_info:
                num_keys_set.add(beatmap["num_keys"])

            # Convert corresponding key value beatmaps into
            for key_count in num_keys_set:
                key_folder = output_beatmap_folder / f"{key_count}k"
                key_folder.mkdir()
                for beatmap in beatmaps_info:
                    count = ord('a')
                    if beatmap["num_keys"] == key_count:
                        one_hot_labels, action_labels, onset_labels = self.generate_labels(beatmap["beatmap"],
                                                                           mel.shape[1],
                                                                           beatmap["num_keys"])
                        if not mel.shape[1] == len(beat_frac) == len(beat_num) == len(action_labels) == len(onset_labels):
                            print(colored(f"Error with generation of labels in {beatmap_folder.name}. Exiting for debugging", "red"))
                            shutil.rmtree(output_beatmap_folder)
                            raise Exception("Error at line 253")

                        np.save((key_folder / f"beatmap_{beatmap["difficulty"]}_{chr(count)}"),
                                {"one_hot": one_hot_labels,
                                 "actions": action_labels, 
                                 "onsets": onset_labels,
                                 "beatmap": beatmap["beatmap"],
                                 "difficulty": beatmap["difficulty"]})
                        
                        if print_json:
                            with open(key_folder / f"beatmap_{beatmap["difficulty"]}_{chr(count)}", "w") as f:
                                json.dump({"actions": action_labels.tolist(),
                                     "onsets": onset_labels,
                                     "beatmap": beatmap["beatmap"].tolist(),
                                     "difficulty": beatmap["difficulty"]}, f, indent=2)
                        
                        count += 1

            if not self.verify(output_beatmap_folder, len(beatmaps_info)):
                print(colored(f"Error in verification of {beatmap_folder.name}", "red"))
                shutil.rmtree(output_beatmap_folder)
                raise Exception("Error when verifying")
            
            print(colored(f"Finished processing {beatmap_folder.name}", "green"))

if __name__ == "__main__":
    osu_path = Path("./preprocess")
    output_path = Path("./postprocess")
    converter = Converter(osu_path, output_path)

    converter.convert()