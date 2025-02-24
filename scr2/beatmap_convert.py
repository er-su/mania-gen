import librosa
import math
from pathlib import Path
import numpy as np

DIFFICULTY_0 = ["ez","beginner","easy","light"]
DIFFICULTY_1 = ["nm","standard","normal"]
DIFFICULTY_2 = ["hard","advanced","hd"]
DIFFICULTY_3 = ["insane","in","another"]
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
                 hop_len=10):
        
        self.beatmap_list_path = beatmap_list_path
        self.output_folder_path = output_folder_path
        self.hop_len = int((hop_len / 1000) * 44100)
        self.n_fft_list = n_fft_list

    #TODO set the beat frac to be one of n number of fractions to minimize noise
    def beat_frac(self, time, offset, bpm):
        beat_frac = ((time - offset) % bpm) / bpm
        return beat_frac
    
    def beat_num(self, time, offset, bpm, meter=4):
        return math.floor((((time - offset) % bpm) / bpm) * meter)

    def convert_audio(self, y, offset, bpm):
        mel = []
        for n_fft_val in self.n_fft_list:
            mel.append(librosa.feature.mel_spectrogram(y=y,
                                                       sr=44100,
                                                       n_fft=n_fft_val,
                                                       hop_length=self.hop_len,
                                                       n_mels=40,
                                                       power=2))
        mel = np.stack(mel, dim=0)
        #Check
        beat_frac = [beat_frac(x * self.hop_len, offset, bpm) for x in range(mel.shape[1])]
        beat_num = [beat_num(x * self.hop_len, offset, bpm) for x in range(mel.shape[1])]

        return mel, beat_frac, beat_num
    
    def parse_osu(self, beatmap_folder, osu_fn):
        osu_path = beatmap_folder / osu_fn

        with open(osu_path, mode='r', enconding="utf-8") as f:
            data = f.read().splitlines()

        # Ensure it's mania
        i = data.index("[General]") + 1
        while data[i] != "" or "[" in data[i]:
            line = data[i].lower()
            if "AudioFilename:" in data[i]:
                audiofile = line.split(' ')[-1]
            
            if "mode:" in line:
                if int(line.split(' ')) != 3:
                    print(f"Beatmap {osu_fn} is not mania")
                    return -1, -1, -1, -1, -1, -1
                else:
                    break

        # Get difficulty
        difficulty = -1
        version = -1
        i = data.index("[Metadata]") + 1
        while data[i] != "" or "[" in data[i]:
            line = data[i].lower()
            if "version:" in line:
                version = i
                break
        
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
        while data[i] != "" or "[" in data[i]:
            line = data[i].lower()
            if "circlesize:" in line:
                num_keys = int(line.split(':')[-1])

        # Get BPM and offset
        # TODO: fix if same bpm but different offset midway thru
        timing_pts = []
        i = data.index("[TimingPoints]") + 1
        while data[i] != "" or "[" in data[i]:
            timing_pts.append(data[i])

        bpms = []
        meters = []
        for tp in timing_pts:
            if float(tp.split(',')[1]) > 0: 
                bpms.append(float(tp.split(',')[1]))
            if int(tp.split(',')[2]) > 0:
                meters.append(int(tp.split(',')[2]))

        if bpms.count(bpms[0]) != len(bpms) or meters.count(meters[0]) != len(meters):
            print(f"More than one meter or BPM in {osu_fn}")
            return -1, -1, -1, -1, -1, -1
        if meters[0] != 4:
            print(f"Multiple meters in {osu_fn}")
            return -1, -1, -1, -1, -1, -1
        
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
        
    def generate_labels(self, obj_list, num_timesteps, num_keys):

        # Obj_list is of size num_timesteps x 3 where each obj is [time, key, type 0-3]
        new_obj = obj_list.copy()
        for obj in new_obj:
            obj[0] = math.round(obj[0] / self.hop_len) # Value now represents index every 10ms

        # Convert to individual actions of shape num_timesteps by num_keys
        # where the values represent the type of action 
        action_array = np.zeros((num_timesteps, num_keys))
        for obj in new_obj:
            time_in_10s, key, note_type = obj
            action_array[int(time_in_10s), int(key)] = note_type


    def convert(self):
        beatmap_list = self.osu_path.iterdir()

        beatmaps_info = []
        for beatmap_folder in beatmap_list:
            for file in beatmap_folder.iterdir():
                if file.suffix == ".osu":
                    beatmap, num_keys, offset, bpm, difficulty, audiofile = self.parse_osu(beatmap_folder, file.name)
                    beatmaps_info.append(
                        {"beatmap": beatmap,
                         "num_keys": num_keys,
                         "offset": offset,
                         "bpm": bpm,
                         "difficulty": difficulty,
                         "audiofile": audiofile,
                        }
                    )


                
