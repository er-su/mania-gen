import numpy as np
import model_func

THRESHOLD = 0.5

def convert_to_hit_objects(predictions, time_step=20, sr=22050):
    short = predictions["short"].numpy()
    long = predictions["long"].numpy()

    num_keys = short.shape[-1]
    hit_objects = []

    for t, short_frame, long_frame in zip(enumerate(short), long):
        time = t * time_step / sr * 1000 # Convert to msec
        for key in range(num_keys):
            press_prob = short_frame[key]
            hold_time = long_frame[key] * 1000

            if press_prob > THRESHOLD:
                hit_obj = {
                    "x": 64 + (key * 128 // num_keys),
                    "y": 192,
                    "time": int(time),
                    "type": 128 if hold_time > 0 else 1,
                    "end_time": int(time + hold_time) if hold_time > 0 else 0
                }

                hit_objects.append(hit_obj)

    return hit_objects

def convert_to_osu(hit_objects):
    with open("result.osu", "w") as f:
        f.write("[HitObjects]")
        for t, object in enumerate(hit_objects):
            if t is 0 and object["type"] is not 128:
                f.write(f"{object["x"]},{object["y"]},{object["time"]},5,0,{object["end_time"]}:0:0:0:0:")
            else:
                f.write(f"{object["x"]},{object["y"]},{object["time"]},{object["type"]},0,{object["end_time"]}:0:0:0:0:")

if __name__ == "__main__":
    model = model_func.load_model("PATH")
    predictions = model_func.inference(model, "INPUT", "INPUT MEL")
    hit_objects = convert_to_hit_objects(predictions)
    convert_to_osu(hit_objects)
