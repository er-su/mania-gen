import numpy as np
import model_func
import math
from data_loader import prepare_input_output

THRESHOLD = 0.5

def convert_to_hit_objects(predictions, time_step=20, sr=22050):
    short = predictions["short"].numpy()[0]
    long = predictions["long"].numpy()[0]

    print(short.shape)
    print(long.shape)
    num_keys = short.shape[-1]
    hit_objects = []

    for t, short_frame in enumerate(short):
        time = t * time_step / sr * 1000 # Convert to msec
        long_frame = long[t]
        for key in range(num_keys):
            press_prob = short_frame[key]
            hold_time = long_frame[key] * 1000

            if press_prob > THRESHOLD:
                hit_obj = {
                    "x": ((key * 512) // 4) + (256 // num_keys),
                    # fix this line"x": 64 + (key * 128 // num_keys),
                    "y": 192,
                    "time": int(time),
                    "type": 128 if hold_time > 0 else 1,
                    "end_time": int(time + hold_time) if hold_time > 0 else 0
                }

                hit_objects.append(hit_obj)

    return hit_objects

def convert_to_osu(hit_objects, output_path="default.osu"):
    with open(output_path, "w") as f:
        f.write("[HitObjects]\n")
        for t, object in enumerate(hit_objects):
            if t == 0 and object["type"] != 128:
                f.write(f"{object["x"]},{object["y"]},{object["time"]},5,0,{object["end_time"]}:0:0:0:0:\n")
            else:
                f.write(f"{object["x"]},{object["y"]},{object["time"]},{object["type"]},0,{object["end_time"]}:0:0:0:0:\n")

if __name__ == "__main__":
    model = model_func.load_model("default_name.keras")
    mel, data, short, long, size = prepare_input_output("../data/processed_beatmaps/1654907 Hoshimachi Suisei - Kakero")
    predictions = model_func.inference(model, data, mel)
    hit_objects = convert_to_hit_objects(predictions)
    convert_to_osu(hit_objects)
