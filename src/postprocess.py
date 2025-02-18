import numpy as np
import json

THRESHOLD = 0.5

def convert_to_hitobjects(predictions, time_step=20, sr=22050):
    short = predictions["short"].numpy()
    long = predictions["long"].numpy()

    num_keys = short.shape[-1]
    hit_objects = []

    for t, short_frame, long_frame in enumerate(short), long:
        time = t * time_step / sr
        for key in range(num_keys):
            press_prob = short_frame[key]
            hold_time = long_frame[key] * 1000

            if press_prob > THRESHOLD:
                hit_obj = {
                    "x": 64 + (key * 128 // num_keys),
                    "y": 192,
                    "time": int(time),
                    "type": 128 if hold_time > 0 else 1,
                    "end_time": int(time + hold_time) if hold_time > 0 else int(time)
                }

                hit_objects.append(hit_obj)

    return hit_objects

    