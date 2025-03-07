import shutil
import gradio as gr
from tqdm import tqdm
from pathlib import Path
from infer import main as infer

model_paths = [
    Path("checkpoints/V1.pt"),
    Path("checkpoints/V2.pt"),
    Path("checkpoints/V3.pt")
]

valid_extensions = (
    ".ogg",
    ".mp3",
    ".wav",
    ".OGG",
    ".MP3",
    ".WAV"
)

def validate_input(audio_file, bpm, offset, artist, title):
    if not audio_file:
        raise gr.Error("Must include audio file")
    
    if not audio_file.endswith(valid_extensions):
        raise gr.Error("Wrong filetype")
    
    if bpm < 0 or bpm is None:
        raise gr.Error("Must include valid bpm")

    if not offset:
        raise gr.Error("Must include offset") 
    
    return

def download():
    return [gr.Button("Generate Beatmap"),
            gr.DownloadButton(label="First Generate a Map!", interactive=False)]

def generate_infers(audio_file, bpm, offset, title, artist, progress=gr.Progress()):
    if not audio_file.endswith(valid_extensions): 
        gr.Error("Not valid audio type")
        return [None, None]
    
    audio_path = Path(audio_file)
    output_files = []
    output_dir = Path("generated_beatmaps")
    output_dir.mkdir(exist_ok=True)

    osz_audio = output_dir / f"audio{audio_path.suffix}"

    shutil.copy(audio_path, osz_audio)

    for i, model in enumerate(progress.tqdm(model_paths, desc="Generating beatmaps...")):
        print(f"Creating {artist} - {title} (mania-gen) [V{i}].osu")
        output = output_dir / f"{artist} - {title} (mania-gen) [V{i}].osu"
        infer(audio_path, bpm, offset, artist=artist, title=title, model_state_dic=model, save_path=output)
        output_files.append(str(output))

    osz_filename = Path(f"{artist} - {title}")
    osz_filename = Path(shutil.make_archive(osz_filename, "zip", output_dir))
    osz_filename = osz_filename.rename(osz_filename.with_suffix(".osz"))

    shutil.rmtree(output_dir)

    return [gr.Button("Generate Beatmp", visible=True),
            gr.DownloadButton(label="Download .osz File", value=osz_filename, interactive=True)]

with gr.Blocks() as ui:
    gr.Markdown("# Beatmap Generator")

    with gr.Column():
        with gr.Row():
            audio_input = gr.Audio(sources="upload", label="Upload Audio File", type="filepath", format="mp3")
            with gr.Column():
                bpm_input = gr.Number(label="BPM")
                offset_input = gr.Number(label="Offset")

        with gr.Row():
            title_input = gr.Textbox(label="Song Title")
            artist_input = gr.Textbox(label="Artist Name")

    with gr.Row():
        process_button = gr.Button("Generate Beatmap", visible=True)
        d1 = gr.DownloadButton(label="First Generate a Map!", interactive=False)

    process_button.click(fn=validate_input, 
                         inputs=[audio_input, bpm_input, offset_input, title_input, artist_input],
                         outputs=None,
                         show_progress="hidden").success(fn=generate_infers,
                                                         inputs=[audio_input, bpm_input, offset_input, title_input, artist_input],
                                                         outputs=[process_button, d1],
                                                         show_progress="full",
                                                         show_progress_on=[process_button, d1],
                                                         trigger_mode="once")
    d1.click(fn=download, inputs=None, outputs=[process_button, d1], show_progress="full")

if __name__ == "__main__":
    ui.launch()