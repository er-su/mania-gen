import torch
import torchaudio
import numpy as np
from pathlib import Path
from pytorch_model import OsuGen
from beatmap_convert import Converter
from pytorch_model import hyperparams

# Given an audio file, bpm, offset, artist name, song title, output dir, and model path
# perform inference on the audio with the model, then generate a .osu file
def main(audio_path: Path,
         bpm,
         offset,
         title="title_here", 
         artist="artist_here",
         user="mania-gen",
         model_state_dic: Path = Path("checkpoints/V3.pt"),
         save_path=None):

    if save_path == None:
        save_path = Path(f"{artist} - {title} ({user}) [placeholder difficulty].osu")

    # Load in converter and model
    converter = Converter(None, None)
    model = OsuGen(hyperparams)
    model.load_state_dict(torch.load(model_state_dic, map_location=torch.device("cpu")))

    bpm = 60000 / bpm
    print(f"Attempting to convert audio file of {audio_path.name}")
    y, sr = torchaudio.load(audio_path)
    y = y.mean(dim=0)

    if sr != 44100:
        y = torchaudio.functional.resample(y, sr, 44100)
    
    # Convert to tensor and pad with batch_size = 1
    mels, beat_fracs, beat_nums = converter.convert_audio(y, offset, bpm)
    mels = torch.as_tensor(np.array(mels)).unsqueeze(dim=0)
    beat_fracs = torch.as_tensor(beat_fracs).unsqueeze(dim=0)
    beat_nums = torch.as_tensor(beat_nums).unsqueeze(dim=0)

    model.eval()
    with torch.inference_mode():
        out = model.old_infer(mels, beat_fracs, beat_nums)
    

    print("Writing beatmap...")
    decoded_beatmap = []

    # Convert back to ms and keys
    for i, action_pred in enumerate(out):
        if action_pred.item() > 0:
            # TODO: Snap to nearest 1/4 beat if within 10 ms of it
            #action = [i * 10]
            action = [snap_to(i * 10, bpm, offset)]
            combo = base_n_decoding(action_pred.item(), num_keys=4)
            for key_vals in combo:
                action.append(key_vals)
            decoded_beatmap.append(action)
    
    # Create an array of the lines that will be written into the .osu file
    hit_objects = []
    is_held = [False, False, False, False]
    held_start_lines = [[],[],[],[]]
    held_end_times = [[],[],[],[]]
    line = 0
    for action in decoded_beatmap:
        time = action[0]
        for col, val in enumerate(action[1:]):
            if val > 0:
                xpos = col * 128 + 64
                if val == 2 and not is_held[col]:
                    held_start_lines[col].append(line)
                    hit_objects.append(f"{xpos},192,{time},128,0,")
                    is_held[col] = True
                    line += 1

                elif val == 2 and is_held[col]:
                    continue
                
                elif val == 3 and is_held[col]:
                    held_end_times[col].append(time)
                    is_held[col] = False

                elif val ==1 and is_held[col]:
                    held_end_times[col].append(time)
                    is_held[col] = False
                    
                else:
                    hit_objects.append(f"{xpos},192,{time},1,0,0:0:0:0:")
                    line += 1 

    # Go back and apply any ends of long notes 
    for col, starts in enumerate(held_start_lines):
        for i, line in enumerate(starts):
            if i >= len(held_end_times[col]):
                split_line = hit_objects[line].split(',')
                hit_objects[line] = f"{split_line[0]},{split_line[1]},{split_line[2]},1,0,0:0:0:0:"
                continue

            endtime = held_end_times[col][i]
            hit_objects[line] = f"{hit_objects[line]}{endtime}:0:0:0:0:"

    suffix = audio_path.suffix

    # Write osu file
    with open(save_path, "w") as f:
        f.write("osu file format v14\n\n")

        f.write("[General]\n")
        f.write(f"AudioFilename: audio{suffix}\n")
        f.write("AudioLeadIn: 0\n")
        f.write("PreviewTime: -1\n")
        f.write("Countdown: 0\n")
        f.write("SampleSet: Normal\n")
        f.write("StackLeniency: 0.7\n")
        f.write("Mode: 3\n")
        f.write("LetterboxInBreaks: 0\n")
        f.write('SpecialStyle: 0\n')
        f.write('WidescreenStoryboard: 1\n\n')

        f.write('[Editor]\n')
        f.write('DistanceSpacing: 0.8\n')
        f.write('BeatDivisor: 4\n')
        f.write('GridSize: 32\n')
        f.write('TimelineZoom: 2.4\n\n')

        f.write('[Metadata]\n')
        f.write(f'Title:{title}\n')
        f.write(f'TitleUnicode:{title}\n')
        f.write(f'Artist:{artist}\n')
        f.write(f'ArtistUnicode:{artist}\n')
        f.write('Creator:mania-gen\n')
        f.write(f'Version:{model_state_dic.stem}\n')
        f.write('Source:\n')
        f.write('Tags:\n')
        f.write('BeatmapID:0\n')
        f.write('BeatmapSetID:-1\n\n')

        f.write('[Difficulty]\n')
        f.write('HPDrainRate:5\n')
        f.write('CircleSize:4\n')
        f.write('OverallDifficulty:5\n')
        f.write('ApproachRate:5\n')
        f.write('SliderMultiplier:1.4\n')
        f.write('SliderTickRate:1\n\n')

        f.write("[Events]\n")
        f.write('//Background and Video events\n')
        f.write('//Break Periods\n')
        f.write('//Storyboard Layer 0 (Background)\n')
        f.write('//Storyboard Layer 1 (Fail)\n')
        f.write('//Storyboard Layer 2 (Pass)\n')
        f.write('//Storyboard Layer 3 (Foreground)\n')
        f.write('//Storyboard Layer 4 (Overlay)\n')
        f.write('//Storyboard Sound Samples\n\n')

        f.write('[TimingPoints]\n')
        f.write(f'{offset},{bpm},4,2,1,40,1,0\n\n')

        f.write('[HitObjects]\n')
        f.write('\n'.join(hit_objects))

# Given a value and the number of keys
# convert a single integer back into
# the combination of key presses and note types
def base_n_decoding(index, num_keys = 4):
    combo = []
    for i in range(num_keys):
        value = (index // (num_keys ** i)) % num_keys
        combo.append(value)

    return combo

def snap_to(time, bpm, offset):
    time_jump = float((60000 / bpm) / 4)
    lower_bound = ((time - offset) // time_jump) * time_jump + offset
    upper_bound = lower_bound + time_jump

    if abs(lower_bound - time) < 15:
        return round(lower_bound)
    
    elif abs(upper_bound - time) < 15:
        return round(upper_bound)
    
    else:
        return time

if __name__ == "__main__":
    #main(Path("testsongs/colorful.mp3"), 192, 615)
    #main(Path("testsongs/owari.mp3"), 201, 1221)
    #main(Path("testsongs/onigiri.mp3"), 210, 84)
    #main(Path("testsongs/renai.ogg"), 156, 378)
    #main(Path("testsongs/lemonade.mp3"), 191, 421)
    #main(Path("testsongs/president.ogg"), 176, 8)
    #main(Path("testsongs/tsukinami.mp3"), 180, 10992)
    #main(Path("testsongs/blue.ogg"), 198.220, 1195)
    #main(Path("testsongs/samba.mp3"), 150, 686)
    main(Path("testsongs/yuukei.mp3"), 129, 596)