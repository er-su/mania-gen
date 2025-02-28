import torch
import random
import numpy as np
from pathlib import Path
from termcolor import colored
from torch.utils.data import Dataset, DataLoader

class OsuDataset(Dataset):
    def __init__(self, postprocess_path: Path, save_path: Path, num_keys=4, difficulty=(4,5), slice=True):
        self.postprocess_path = postprocess_path
        self.save_path = save_path
        self.num_keys = num_keys
        self.difficulty = difficulty
        self.collector = Collector(postprocess_path, save_path)
        if slice:
            self.mels, self.beat_fracs, self.beat_nums, self.actions, self.onsets = self.collector.collect_from_file(slice_size=3000, num_keys=num_keys, difficulty=difficulty)
        else:
            self.mels, self.beat_fracs, self.beat_nums, self.actions, self.onsets = self.collector.collect_from_file(num_keys=num_keys, difficulty=difficulty)

    def __len__(self):
        return len(self.mels)
    
    def __getitem__(self, index):
        mels = self.mels[index]
        beat_fracs = self.beat_fracs[index]
        beat_nums = self.beat_nums[index]
        onsets = self.onsets[index]
        actions = self.actions[index]

        return mels, beat_fracs, beat_nums, onsets, actions

class Collector:

    def __init__(self, postprocess_path: Path, save_path: Path):
        self.postprocess_path = postprocess_path
        self.save_path = save_path


    def collect_full(self, num_keys=4, difficulty=(4,5), save=True):
        mel_tensors = []
        beat_frac_tensors = []
        beat_num_tensors = []
        actions_tensors = []
        onsets_tensors = []

        beatmap_list = self.postprocess_path.iterdir()
        for beatmap_folder in beatmap_list:
            print(f"Attempting to collect in {beatmap_folder.name}")
            key_folder = beatmap_folder / f"{num_keys}k"
            audiofile = beatmap_folder / "audio_features.npy"

            # Verify that there is 4k and features exist
            if not key_folder.exists():
                print(colored(f"{beatmap_folder.name} does not have {difficulty}k folder", "yellow"))
                continue

            if not audiofile.exists():
                print(colored(f"{beatmap_folder.name} does not have a audio_features file. Skipping", "red"))
                continue
            
            # Load in audio features as dict of "mel", "beat_frac", and "beat_num"
            try:
                audio_features = np.load(audiofile, allow_pickle=True).item()

            except OSError as err:
                print(colored(f"Unable to load audio for {beatmap_folder.name}. Skipping", "red"))
                continue

            mel_tensor = torch.as_tensor(audio_features["mel"])
            beat_frac_tensor = torch.as_tensor(audio_features["beat_frac"])
            beat_num_tensor = torch.as_tensor(audio_features["beat_num"])
            
            # Load in valid beatmaps
            for key_beatmap in key_folder.iterdir():
                if not int(key_beatmap.name.split('_')[1]) in difficulty:
                    #print(f"{key_beatmap.name} is not correct difficulty")
                    continue
                
                print(f"{key_beatmap.name} found in {beatmap_folder.name}. Collecting...")

                try:
                    labels = np.load(key_beatmap, allow_pickle=True).item()
                except OSError as err:
                    print(colored(f"Unable to load beatmap {key_beatmap.name}. Skipping", "red"))
                    continue
                
                actions_tensor = torch.from_numpy(np.array(labels["actions"]))
                onsets_tensor = torch.as_tensor(labels["onsets"])

                assert(actions_tensor.shape[0] == onsets_tensor.shape[0] == mel_tensor.shape[1] 
                       == beat_frac_tensor.shape[0] == beat_num_tensor.shape[0])
                
                mel_tensors.append(mel_tensor)
                beat_frac_tensors.append(beat_frac_tensor)
                beat_num_tensors.append(beat_num_tensor)
                actions_tensors.append(actions_tensor)
                onsets_tensors.append(onsets_tensor)

        assert(len(mel_tensors) == len(actions_tensors))

        if save:
            difficulty_fn = ""
            for dif in difficulty:
                difficulty_fn += f"_{dif}"
            save_fn = f"full_{num_keys}k{difficulty_fn}.pt"

            if (self.save_path / save_fn).exists():
                print(f"{save_fn} already exists. Overwritting...")
                (self.save_path / save_fn).unlink()

            torch.save({"mel_tensors": mel_tensors,
                        "beat_frac_tensors": beat_frac_tensors,
                        "beat_num_tensors": beat_num_tensors,
                        "actions_tensors": actions_tensors,
                        "onsets_tensors": onsets_tensors}, self.save_path / save_fn)
            
        print(colored(f"Found and collected {len(mel_tensors)} beatmaps", "green"))

        return mel_tensors, beat_frac_tensors, beat_num_tensors, actions_tensors, onsets_tensors
                

    def collect_slice(self, slice_size=3000, num_keys=4, difficulty=(4,5), save=True):
        mel_tensors = []
        beat_frac_tensors = []
        beat_num_tensors = []
        actions_tensors = []
        onsets_tensors = []

        beatmap_list = self.postprocess_path.iterdir()
        for beatmap_folder in beatmap_list:
            print(f"Attempting to collect in {beatmap_folder.name}")
            key_folder = beatmap_folder / f"{num_keys}k"
            audiofile = beatmap_folder / "audio_features.npy"

            # Verify that there is 4k and features exist
            if not key_folder.exists():
                print(colored(f"{beatmap_folder.name} does not have {difficulty}k folder", "yellow"))
                continue

            if not audiofile.exists():
                print(colored(f"{beatmap_folder.name} does not have a audio_features file. Skipping", "red"))
                continue
            
            # Load in audio features as dict of "mel", "beat_frac", and "beat_num"
            try:
                audio_features = np.load(audiofile, allow_pickle=True).item()

            except OSError as err:
                print(colored(f"Unable to load audio for {beatmap_folder.name}. Skipping", "red"))
                continue

            full_mel_tensor = torch.as_tensor(audio_features["mel"])
            full_beat_frac_tensor = torch.as_tensor(audio_features["beat_frac"])
            full_beat_num_tensor = torch.as_tensor(audio_features["beat_num"])
            
            # Load in valid beatmaps
            for key_beatmap in key_folder.iterdir():
                if not int(key_beatmap.name.split('_')[1]) in difficulty:
                    print(f"{key_beatmap.name} is not correct difficulty")
                    continue

                try:
                    labels = np.load(key_beatmap, allow_pickle=True).item()
                except OSError as err:
                    print(colored(f"Unable to load beatmap {key_beatmap.name}. Skipping...", "red"))
                    continue
                
                actions_tensor = torch.from_numpy(np.array(labels["actions"]))
                onsets_tensor = torch.as_tensor(labels["onsets"])

                assert(actions_tensor.shape[0] == onsets_tensor.shape[0] == full_mel_tensor.shape[1] == full_beat_frac_tensor.shape[0] == full_beat_num_tensor.shape[0])
                
                # Take random slice
                if full_mel_tensor.shape[1] < slice_size:
                    raise IndexError(f"{beatmap_folder.name} is shorter than slice size")

                rand_start = random.randint(0, full_mel_tensor.shape[1] - slice_size - 1)
                mel_tensor = full_mel_tensor[:, rand_start : rand_start + slice_size, :]
                beat_frac_tensor = full_beat_frac_tensor[rand_start : rand_start + slice_size]
                beat_num_tensor = full_beat_num_tensor[rand_start : rand_start + slice_size]
                actions_tensor = actions_tensor[rand_start : rand_start + slice_size]
                onsets_tensor = onsets_tensor[rand_start : rand_start + slice_size]

                mel_tensors.append(mel_tensor)
                beat_frac_tensors.append(beat_frac_tensor)
                beat_num_tensors.append(beat_num_tensor)
                actions_tensors.append(actions_tensor)
                onsets_tensors.append(onsets_tensor)

        assert(len(mel_tensors) == len(actions_tensors))

        if save:

            difficulty_fn = ""
            for dif in difficulty:
                difficulty_fn += f"_{dif}"
            save_fn = f"sliced_{num_keys}k{difficulty_fn}.pt"

            if (self.save_path / save_fn).exists():
                print(f"{save_fn} already exists. Overwritting...")
                (self.save_path / save_fn).unlink()
        
            torch.save({"mel_tensors": mel_tensors,
                        "beat_frac_tensors": beat_frac_tensors,
                        "beat_num_tensors": beat_num_tensors,
                        "actions_tensors": actions_tensors,
                        "onsets_tensors": onsets_tensors}, self.save_path / save_fn)
            
        print(colored(f"Found and collected {len(mel_tensors)} beatmaps", "green"))

        return mel_tensors, beat_frac_tensors, beat_num_tensors, actions_tensors, onsets_tensors
    
    def collect_from_file(self, slice_size=0, num_keys=4, difficulty=(4,5)):
        difficulty_fn = ""
        sliced = "sliced" if slice_size > 0 else "full"
        for dif in difficulty:
            difficulty_fn += f"_{dif}"
        save_fn = f"{sliced}_{num_keys}k{difficulty_fn}.pt"
        savefile_path = self.save_path / save_fn

        # Check to see if it doesn't alr exist
        if not savefile_path.exists():
            print(f"{save_fn} doesn not exist. Creating...")
            if sliced == "sliced":
                return self.collect_slice(slice_size=slice_size,
                                          num_keys=num_keys, 
                                          difficulty=difficulty,
                                          save=True)
            else:
                return self.collect_full(num_keys=num_keys,
                                         difficulty=difficulty,
                                         save=True)
        
        print(f"Pre-exsting {savefile_path.name} found.")
        dict = torch.load(savefile_path)
        return dict["mel_tensors"], dict["beat_frac_tensors"], dict["beat_num_tensors"], dict["actions_tensors"], dict["onsets_tensors"]

if __name__ == "__main__":
    #collector = Collector(Path("postprocess"), Path("saved"))
    #mels, beat_fracs, beat_nums, actions, onsets = collector.collect_full(difficulty=(3,4,5))

    osudataset = OsuDataset(Path("postprocess"), Path("saved"), difficulty=(3,4,5))

    #loader = DataLoader(osudataset, batch_size=10, shuffle=True)

    #mels, beat_fracs, beat_nums, onsets, actions = next(iter(loader))
    #print(f"Mels shape {mels.size()}")
    #print(f"Beat frac shape {beat_fracs.size()}")
    #print(f"Beat nums shape {beat_nums.size()}")
    #print(f"onsets shape {onsets.size()}")
    #print(f"actions shape {actions.size()}")