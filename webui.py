import html
import json
import os
import sys
import threading
import time
import warnings
import tempfile
import zipfile
import re
import shutil
import traceback

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(
    description="IndexTTS WebUI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
parser.add_argument("--gui_seg_tokens", type=int, default=120, help="GUI: Max tokens per generation segment")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt"
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr
from indextts.infer_v2 import IndexTTS2
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="Auto")
MODE = 'local'

# Initialize TTS model with error handling
def initialize_tts_model():
    try:
        print("Initializing IndexTTS2 model...")
        tts = IndexTTS2(
            model_dir=cmd_args.model_dir,
            cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
            use_fp16=cmd_args.fp16,
            use_deepspeed=cmd_args.deepspeed,
            use_cuda_kernel=cmd_args.cuda_kernel,
        )
        print("Model initialized successfully!")
        return tts
    except Exception as e:
        print(f"Error initializing model: {e}")
        traceback.print_exc()
        return None

tts = initialize_tts_model()
if tts is None:
    print("Failed to initialize TTS model. Exiting.")
    sys.exit(1)

# 支持的语言列表
LANGUAGES = {
    "中文": "zh_CN",
    "English": "en_US"
}
EMO_CHOICES_ALL = [i18n("与音色参考音频相同"),
                i18n("使用情感参考音频"),
                i18n("使用情感向量控制"),
                i18n("使用情感描述文本控制")]
EMO_CHOICES_OFFICIAL = EMO_CHOICES_ALL[:-1]  # skip experimental features

os.makedirs("outputs/tasks",exist_ok=True)
os.makedirs("prompts",exist_ok=True)

MAX_LENGTH_TO_USE_SPEED = 70
example_cases = []
if os.path.exists("examples/cases.jsonl"):
    with open("examples/cases.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            if example.get("emo_audio",None):
                emo_audio_path = os.path.join("examples",example["emo_audio"])
            else:
                emo_audio_path = None

            example_cases.append([os.path.join("examples", example.get("prompt_audio", "sample_prompt.wav")),
                                EMO_CHOICES_ALL[example.get("emo_mode",0)],
                                example.get("text"),
                                emo_audio_path,
                                example.get("emo_weight",1.0),
                                example.get("emo_text",""),
                                example.get("emo_vec_1",0),
                                example.get("emo_vec_2",0),
                                example.get("emo_vec_3",0),
                                example.get("emo_vec_4",0),
                                example.get("emo_vec_5",0),
                                example.get("emo_vec_6",0),
                                example.get("emo_vec_7",0),
                                example.get("emo_vec_8",0),
                                ])

def get_example_cases(include_experimental = False):
    if include_experimental:
        return example_cases  # show every example

    # exclude emotion control mode 3 (emotion from text description)
    return [x for x in example_cases if x[1] != EMO_CHOICES_ALL[3]]

def gen_single(emo_control_method,prompt, text,
               emo_ref_path, emo_weight,
               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
               emo_text,emo_random,
               max_text_tokens_per_segment=120,
                *args, progress=gr.Progress()):
    try:
        output_path = None
        if not output_path:
            output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
        # set gradio progress
        tts.gr_progress = progress
        do_sample, top_p, top_k, temperature, \
            length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
        kwargs = {
            "do_sample": bool(do_sample),
            "top_p": float(top_p),
            "top_k": int(top_k) if int(top_k) > 0 else None,
            "temperature": float(temperature),
            "length_penalty": float(length_penalty),
            "num_beams": num_beams,
            "repetition_penalty": float(repetition_penalty),
            "max_mel_tokens": int(max_mel_tokens),
        }
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value
        if emo_control_method == 0:  # emotion from speaker
            emo_ref_path = None  # remove external reference audio
        if emo_control_method == 1:  # emotion from reference audio
            pass
        if emo_control_method == 2:  # emotion from custom vectors
            vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
            vec = tts.normalize_emo_vec(vec, apply_bias=True)
        else:
            # don't use the emotion vector inputs for the other modes
            vec = None

        if emo_text == "":
            # erase empty emotion descriptions; `infer()` will then automatically use the main prompt
            emo_text = None

        print(f"Emo control mode:{emo_control_method},weight:{emo_weight},vec:{vec}")
        output = tts.infer(spk_audio_prompt=prompt, text=text,
                           output_path=output_path,
                           emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                           emo_vector=vec,
                           use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                           verbose=cmd_args.verbose,
                           max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                           **kwargs)
        return gr.update(value=output,visible=True)
    except Exception as e:
        print(f"Error in gen_single: {e}")
        traceback.print_exc()
        return gr.update(value=None, visible=True)

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button

def create_warning_message(warning_text):
    return gr.HTML(f"<div style=\"padding: 0.5em 0.8em; border-radius: 0.5em; background: #ffa87d; color: #000; font-weight: bold\">{html.escape(warning_text)}</div>")

def create_experimental_warning_message():
    return create_warning_message(i18n('提示：此功能为实验版，结果尚不稳定，我们正在持续优化中。'))

# Multi-speaker mode functions
def parse_multi_speaker_script(script_text):
    pattern = r'\[([^\]]+)\]\{([^}]*)\}'
    matches = re.findall(pattern, script_text)
    return [(speaker, text.strip()) for speaker, text in matches]

def generate_multi_speaker_audio(num_speakers, speaker_audios, speaker_names, script_text,
                                # Emotion control parameters
                                emo_control_method, emo_ref_path, emo_weight,
                                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                                emo_text, emo_random,
                                # Advanced parameters
                                do_sample, top_p, top_k, temperature,
                                length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                                max_text_tokens_per_segment,
                                progress=gr.Progress()):
    try:
        # Create mapping of speaker names to audio files
        speaker_map = {}
        for i in range(num_speakers):
            name = speaker_names[i]
            audio = speaker_audios[i]
            if name and audio:
                speaker_map[name] = audio
        
        # Parse script
        segments = parse_multi_speaker_script(script_text)
        
        if not segments:
            raise ValueError("No valid script segments found. Please use [SpeakerName]{Text} format.")
        
        # Create temporary directory for generated files
        temp_dir = tempfile.mkdtemp()
        generated_files = []
        
        # Prepare emotion parameters
        kwargs = {
            "do_sample": bool(do_sample),
            "top_p": float(top_p),
            "top_k": int(top_k) if int(top_k) > 0 else None,
            "temperature": float(temperature),
            "length_penalty": float(length_penalty),
            "num_beams": num_beams,
            "repetition_penalty": float(repetition_penalty),
            "max_mel_tokens": int(max_mel_tokens),
        }
        
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value
        if emo_control_method == 0:  # emotion from speaker
            emo_ref_path = None  # remove external reference audio
        if emo_control_method == 1:  # emotion from reference audio
            pass
        if emo_control_method == 2:  # emotion from custom vectors
            vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
            vec = tts.normalize_emo_vec(vec, apply_bias=True)
        else:
            # don't use the emotion vector inputs for the other modes
            vec = None

        if emo_text == "":
            # erase empty emotion descriptions; `infer()` will then automatically use the main prompt
            emo_text = None
        
        # Generate audio for each segment
        total_segments = len(segments)
        for i, (speaker, text) in enumerate(segments):
            progress(i / total_segments, desc=f"Generating segment {i+1}/{total_segments} for speaker: {speaker}")
            
            if speaker not in speaker_map:
                print(f"Warning: Speaker '{speaker}' not found in speaker list. Skipping.")
                continue
                
            output_path = os.path.join(temp_dir, f"{speaker}_{i}.wav")
            
            # Generate audio using the provided parameters
            tts.infer(
                spk_audio_prompt=speaker_map[speaker],
                text=text,
                output_path=output_path,
                emo_audio_prompt=emo_ref_path, 
                emo_alpha=emo_weight,
                emo_vector=vec,
                use_emo_text=(emo_control_method==3), 
                emo_text=emo_text,
                use_random=emo_random,
                verbose=cmd_args.verbose,
                max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                **kwargs
            )
            generated_files.append(output_path)
        
        if not generated_files:
            raise ValueError("No audio files were generated. Please check your speaker names and script format.")
        
        # Create zip file
        zip_path = os.path.join(temp_dir, "multi_speaker_audios.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in generated_files:
                zipf.write(file, os.path.basename(file))
        
        return zip_path
    except Exception as e:
        print(f"Error in generate_multi_speaker_audio: {e}")
        traceback.print_exc()
        return None

# Create speaker UI components (max 10 speakers)
def create_speaker_ui_components(max_speakers=10):
    speaker_audios = []
    speaker_names = []
    speaker_rows = []
    
    for i in range(max_speakers):
        with gr.Row(visible=(i < 2)) as speaker_row:  # Default to visible for first 2 speakers
            speaker_audio = gr.Audio(label=f"Speaker {i+1} Reference Audio", type="filepath")
            speaker_name = gr.Textbox(label=f"Speaker {i+1} Name", placeholder=f"Enter name for speaker {i+1}")
            speaker_audios.append(speaker_audio)
            speaker_names.append(speaker_name)
            speaker_rows.append(speaker_row)
    
    return speaker_audios, speaker_names, speaker_rows

# Create emotion control components for multi-speaker mode
def create_emotion_control_components_multi():
    # Emotion control method
    emo_control_method_multi = gr.Radio(
        choices=EMO_CHOICES_OFFICIAL,
        type="index",
        value=EMO_CHOICES_OFFICIAL[0],
        label=i18n("情感控制方式")
    )
    
    # Hidden radio with all choices (for dataset examples)
    emo_control_method_all_multi = gr.Radio(
        choices=EMO_CHOICES_ALL,
        type="index",
        value=EMO_CHOICES_ALL[0],
        label=i18n("情感控制方式"),
        visible=False
    )
    
    # Emotion reference audio group
    with gr.Group(visible=False) as emotion_reference_group_multi:
        with gr.Row():
            emo_upload_multi = gr.Audio(label=i18n("上传情感参考音频"), type="filepath")
    
    # Emotion random sampling
    with gr.Row(visible=False) as emotion_randomize_group_multi:
        emo_random_multi = gr.Checkbox(label=i18n("情感随机采样"), value=False)
    
    # Emotion vector control
    with gr.Group(visible=False) as emotion_vector_group_multi:
        with gr.Row():
            with gr.Column():
                vec1_multi = gr.Slider(label=i18n("喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                vec2_multi = gr.Slider(label=i18n("怒"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                vec3_multi = gr.Slider(label=i18n("哀"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                vec4_multi = gr.Slider(label=i18n("惧"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
            with gr.Column():
                vec5_multi = gr.Slider(label=i18n("厌恶"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                vec6_multi = gr.Slider(label=i18n("低落"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                vec7_multi = gr.Slider(label=i18n("惊喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                vec8_multi = gr.Slider(label=i18n("平静"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
    
    # Emotion text control
    with gr.Group(visible=False) as emo_text_group_multi:
        create_experimental_warning_message()
        with gr.Row():
            emo_text_multi = gr.Textbox(
                label=i18n("情感描述文本"),
                placeholder=i18n("请输入情绪描述（或留空以自动使用目标文本作为情绪描述）"),
                value="",
                info=i18n("例如：委屈巴巴、危险在悄悄逼近")
            )
    
    # Emotion weight
    with gr.Row(visible=False) as emo_weight_group_multi:
        emo_weight_multi = gr.Slider(label=i18n("情感权重"), minimum=0.0, maximum=1.0, value=0.65, step=0.01)
    
    return (
        emo_control_method_multi, emo_control_method_all_multi,
        emotion_reference_group_multi, emo_upload_multi,
        emotion_randomize_group_multi, emo_random_multi,
        emotion_vector_group_multi, vec1_multi, vec2_multi, vec3_multi, vec4_multi,
        vec5_multi, vec6_multi, vec7_multi, vec8_multi,
        emo_text_group_multi, emo_text_multi,
        emo_weight_group_multi, emo_weight_multi
    )

# Create advanced settings components for multi-speaker mode
def create_advanced_settings_components_multi():
    with gr.Accordion(i18n("高级生成参数设置"), open=False, visible=True) as advanced_settings_group_multi:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(f"**{i18n('GPT2 采样设置')}** _{i18n('参数会影响音频多样性和生成速度详见')} [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)._")
                with gr.Row():
                    do_sample_multi = gr.Checkbox(label="do_sample", value=True, info=i18n("是否进行采样"))
                    temperature_multi = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                with gr.Row():
                    top_p_multi = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                    top_k_multi = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                    num_beams_multi = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                with gr.Row():
                    repetition_penalty_multi = gr.Number(label="repetition_penalty", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                    length_penalty_multi = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                max_mel_tokens_multi = gr.Slider(
                    label="max_mel_tokens", 
                    value=1500, 
                    minimum=50, 
                    maximum=tts.cfg.gpt.max_mel_tokens, 
                    step=10, 
                    info=i18n("生成Token最大数量，过小导致音频被截断")
                )
            with gr.Column(scale=2):
                gr.Markdown(f'**{i18n("分句设置")}** _{i18n("参数会影响音频质量和生成速度")}_')
                with gr.Row():
                    initial_value = max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                    max_text_tokens_per_segment_multi = gr.Slider(
                        label=i18n("分句最大Token数"), 
                        value=initial_value, 
                        minimum=20, 
                        maximum=tts.cfg.gpt.max_text_tokens, 
                        step=2,
                        info=i18n("建议80~200之间，值越大，分句越长；值越小，分句越碎；过小过大都可能导致音频质量不高")
                    )
                with gr.Accordion(i18n("预览分句结果"), open=True) as segments_settings_multi:
                    segments_preview_multi = gr.Dataframe(
                        headers=[i18n("序号"), i18n("分句内容"), i18n("Token数")],
                        wrap=True,
                    )
    
    advanced_params_multi = [
        do_sample_multi, top_p_multi, top_k_multi, temperature_multi,
        length_penalty_multi, num_beams_multi, repetition_penalty_multi, max_mel_tokens_multi,
    ]
    
    return (
        advanced_settings_group_multi,
        do_sample_multi, top_p_multi, top_k_multi, temperature_multi,
        length_penalty_multi, num_beams_multi, repetition_penalty_multi, max_mel_tokens_multi,
        max_text_tokens_per_segment_multi,
        segments_settings_multi, segments_preview_multi,
        advanced_params_multi
    )

# Function to generate audio for text chunks
def generate_text_chunks(prompt, text_chunks, 
                         # Emotion control parameters
                         emo_control_method, emo_ref_path, emo_weight,
                         vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                         emo_text, emo_random,
                         # Advanced parameters
                         do_sample, top_p, top_k, temperature,
                         length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                         max_text_tokens_per_segment,
                         progress=gr.Progress()):
    try:
        # Filter out empty chunks
        valid_chunks = [(i, text) for i, text in enumerate(text_chunks) if text and text.strip()]
        
        if not valid_chunks:
            raise ValueError("No valid text chunks found. Please enter some text.")
        
        # Create temporary directory for generated files
        temp_dir = tempfile.mkdtemp()
        generated_files = []
        
        # Prepare emotion parameters
        kwargs = {
            "do_sample": bool(do_sample),
            "top_p": float(top_p),
            "top_k": int(top_k) if int(top_k) > 0 else None,
            "temperature": float(temperature),
            "length_penalty": float(length_penalty),
            "num_beams": num_beams,
            "repetition_penalty": float(repetition_penalty),
            "max_mel_tokens": int(max_mel_tokens),
        }
        
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value
        if emo_control_method == 0:  # emotion from speaker
            emo_ref_path = None  # remove external reference audio
        if emo_control_method == 1:  # emotion from reference audio
            pass
        if emo_control_method == 2:  # emotion from custom vectors
            vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
            vec = tts.normalize_emo_vec(vec, apply_bias=True)
        else:
            # don't use the emotion vector inputs for the other modes
            vec = None

        if emo_text == "":
            # erase empty emotion descriptions; `infer()` will then automatically use the main prompt
            emo_text = None
        
        # Generate audio for each chunk
        total_chunks = len(valid_chunks)
        for idx, (original_idx, text) in enumerate(valid_chunks):
            progress(idx / total_chunks, desc=f"Generating chunk {idx+1}/{total_chunks}")
            
            output_path = os.path.join(temp_dir, f"chunk_{original_idx+1}.wav")
            
            # Generate audio using the provided parameters
            tts.infer(
                spk_audio_prompt=prompt,
                text=text,
                output_path=output_path,
                emo_audio_prompt=emo_ref_path, 
                emo_alpha=emo_weight,
                emo_vector=vec,
                use_emo_text=(emo_control_method==3), 
                emo_text=emo_text,
                use_random=emo_random,
                verbose=cmd_args.verbose,
                max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                **kwargs
            )
            generated_files.append(output_path)
        
        # Create zip file
        zip_path = os.path.join(temp_dir, "text_chunks_audios.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in generated_files:
                zipf.write(file, os.path.basename(file))
        
        return zip_path
    except Exception as e:
        print(f"Error in generate_text_chunks: {e}")
        traceback.print_exc()
        return None

with gr.Blocks(title="IndexTTS Demo") as demo:
    mutex = threading.Lock()
    gr.HTML('''
    <h2><center>IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech</h2>
<p align="center">
<a href='https://arxiv.org/abs/2506.21619'><img src='https://img.shields.io/badge/ArXiv-2506.21619-red'></a>
</p>
    ''')

    with gr.Tab(i18n("音频生成")):
        with gr.Row():
            os.makedirs("prompts",exist_ok=True)
            prompt_audio = gr.Audio(label=i18n("音色参考音频"),key="prompt_audio",
                                    sources=["upload","microphone"],type="filepath")
            prompt_list = os.listdir("prompts")
            default = ''
            if prompt_list:
                default = prompt_list[0]
            with gr.Column():
                # Text chunks container
                with gr.Column():
                    gr.Markdown("### Text Chunks")
                    gr.Markdown("Enter your text in chunks below. Each chunk will be processed individually and numbered sequentially.")
                    
                    # Initial text chunk
                    text_chunk_1 = gr.TextArea(label="Chunk 1", placeholder="Enter first chunk of text here...", lines=4)
                    
                    # Container for additional chunks
                    with gr.Column() as additional_chunks_container:
                        pass
                    
                    # Buttons to add/remove chunks
                    with gr.Row():
                        add_chunk_btn = gr.Button("Add Text Chunk")
                        remove_chunk_btn = gr.Button("Remove Last Chunk", interactive=False)
                    
                    # Hidden state to track number of chunks
                    num_chunks = gr.State(value=1)
                    
                    # Generation button
                    gen_button = gr.Button(i18n("生成语音"), key="gen_button", interactive=True, variant="primary")
                
                # Output for zip file
                output_zip = gr.File(label="Download Generated Audio (Zip)", visible=False)

        experimental_checkbox = gr.Checkbox(label=i18n("显示实验功能"), value=False)

        with gr.Accordion(i18n("功能设置")):
            # 情感控制选项部分
            with gr.Row():
                emo_control_method = gr.Radio(
                    choices=EMO_CHOICES_OFFICIAL,
                    type="index",
                    value=EMO_CHOICES_OFFICIAL[0],label=i18n("情感控制方式"))
                # we MUST have an extra, INVISIBLE list of *all* emotion control
                # methods so that gr.Dataset() can fetch ALL control mode labels!
                # otherwise, the gr.Dataset()'s experimental labels would be empty!
                emo_control_method_all = gr.Radio(
                    choices=EMO_CHOICES_ALL,
                    type="index",
                    value=EMO_CHOICES_ALL[0], label=i18n("情感控制方式"),
                    visible=False)  # do not render
        # 情感参考音频部分
        with gr.Group(visible=False) as emotion_reference_group:
            with gr.Row():
                emo_upload = gr.Audio(label=i18n("上传情感参考音频"), type="filepath")

        # 情感随机采样
        with gr.Row(visible=False) as emotion_randomize_group:
            emo_random = gr.Checkbox(label=i18n("情感随机采样"), value=False)

        # 情感向量控制部分
        with gr.Group(visible=False) as emotion_vector_group:
            with gr.Row():
                with gr.Column():
                    vec1 = gr.Slider(label=i18n("喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec2 = gr.Slider(label=i18n("怒"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec3 = gr.Slider(label=i18n("哀"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec4 = gr.Slider(label=i18n("惧"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                with gr.Column():
                    vec5 = gr.Slider(label=i18n("厌恶"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec6 = gr.Slider(label=i18n("低落"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec7 = gr.Slider(label=i18n("惊喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec8 = gr.Slider(label=i18n("平静"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)

        with gr.Group(visible=False) as emo_text_group:
            create_experimental_warning_message()
            with gr.Row():
                emo_text = gr.Textbox(label=i18n("情感描述文本"),
                                      placeholder=i18n("请输入情绪描述（或留空以自动使用目标文本作为情绪描述）"),
                                      value="",
                                      info=i18n("例如：委屈巴巴、危险在悄悄逼近"))

        with gr.Row(visible=False) as emo_weight_group:
            emo_weight = gr.Slider(label=i18n("情感权重"), minimum=0.0, maximum=1.0, value=0.65, step=0.01)

        with gr.Accordion(i18n("高级生成参数设置"), open=False, visible=True) as advanced_settings_group:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"**{i18n('GPT2 采样设置')}** _{i18n('参数会影响音频多样性和生成速度详见')} [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)._")
                    with gr.Row():
                        do_sample = gr.Checkbox(label="do_sample", value=True, info=i18n("是否进行采样"))
                        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                    with gr.Row():
                        top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                        top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                        num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                    with gr.Row():
                        repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                        length_penalty = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                    max_mel_tokens = gr.Slider(label="max_mel_tokens", value=1500, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10, info=i18n("生成Token最大数量，过小导致音频被截断"), key="max_mel_tokens")
                with gr.Column(scale=2):
                    gr.Markdown(f'**{i18n("分句设置")}** _{i18n("参数会影响音频质量和生成速度")}_')
                    with gr.Row():
                        initial_value = max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                        max_text_tokens_per_segment = gr.Slider(
                            label=i18n("分句最大Token数"), value=initial_value, minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2, key="max_text_tokens_per_segment",
                            info=i18n("建议80~200之间，值越大，分句越长；值越小，分句越碎；过小过大都可能导致音频质量不高"),
                        )
                    with gr.Accordion(i18n("预览分句结果"), open=True) as segments_settings:
                        segments_preview = gr.Dataframe(
                            headers=[i18n("序号"), i18n("分句内容"), i18n("Token数")],
                            key="segments_preview",
                            wrap=True,
                        )
            advanced_params = [
                do_sample, top_p, top_k, temperature,
                length_penalty, num_beams, repetition_penalty, max_mel_tokens,
            ]

        # we must use `gr.Dataset` to support dynamic UI rewrites, since `gr.Examples`
        # binds tightly to UI and always restores the initial state of all components,
        # such as the list of available choices in emo_control_method.
        example_table = gr.Dataset(label="Examples",
            samples_per_page=20,
            samples=get_example_cases(include_experimental=False),
            type="values",
            # these components are NOT "connected". it just reads the column labels/available
            # states from them, so we MUST link to the "all options" versions of all components,
            # such as `emo_control_method_all` (to be able to see EXPERIMENTAL text labels)!
            components=[prompt_audio,
                        emo_control_method_all,  # important: support all mode labels!
                        text_chunk_1,  # Use the first text chunk for examples
                        emo_upload,
                        emo_weight,
                        emo_text,
                        vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        )

    def on_example_click(example):
        print(f"Example clicked: ({len(example)} values) = {example!r}")
        return (
            gr.update(value=example[0]),
            gr.update(value=example[1]),
            gr.update(value=example[2]),  # Set the first text chunk
            gr.update(value=example[3]),
            gr.update(value=example[4]),
            gr.update(value=example[5]),
            gr.update(value=example[6]),
            gr.update(value=example[7]),
            gr.update(value=example[8]),
            gr.update(value=example[9]),
            gr.update(value=example[10]),
            gr.update(value=example[11]),
            gr.update(value=example[12]),
            gr.update(value=example[13]),
        )

    # click() event works on both desktop and mobile UI
    example_table.click(on_example_click,
                        inputs=[example_table],
                        outputs=[prompt_audio,
                                 emo_control_method,
                                 text_chunk_1,
                                 emo_upload,
                                 emo_weight,
                                 emo_text,
                                 vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
    )

    def on_input_text_change(text, max_text_tokens_per_segment):
        if text and len(text) > 0:
            text_tokens_list = tts.tokenizer.tokenize(text)

            segments = tts.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=int(max_text_tokens_per_segment))
            data = []
            for i, s in enumerate(segments):
                segment_str = ''.join(s)
                tokens_count = len(s)
                data.append([i, segment_str, tokens_count])
            return {
                segments_preview: gr.update(value=data, visible=True, type="array"),
            }
        else:
            df = pd.DataFrame([], columns=[i18n("序号"), i18n("分句内容"), i18n("Token数")])
            return {
                segments_preview: gr.update(value=df),
            }

    def on_method_change(emo_control_method):
        if emo_control_method == 1:  # emotion reference audio
            return (gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 2:  # emotion vectors
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 3:  # emotion text description
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True)
                    )
        else:  # 0: same as speaker voice
            return (gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                    )

    emo_control_method.change(on_method_change,
        inputs=[emo_control_method],
        outputs=[emotion_reference_group,
                 emotion_randomize_group,
                 emotion_vector_group,
                 emo_text_group,
                 emo_weight_group]
    )

    def on_experimental_change(is_experimental, current_mode_index):
        # 切换情感控制选项
        new_choices = EMO_CHOICES_ALL if is_experimental else EMO_CHOICES_OFFICIAL
        # if their current mode selection doesn't exist in new choices, reset to 0.
        # we don't verify that OLD index means the same in NEW list, since we KNOW it does.
        new_index = current_mode_index if current_mode_index < len(new_choices) else 0

        return (
            gr.update(choices=new_choices, value=new_choices[new_index]),
            gr.update(samples=get_example_cases(include_experimental=is_experimental)),
        )

    experimental_checkbox.change(
        on_experimental_change,
        inputs=[experimental_checkbox, emo_control_method],
        outputs=[emo_control_method, example_table]
    )

    # Function to add a new text chunk
    def add_text_chunk(num_chunks):
        new_num = num_chunks + 1
        if new_num > 20:  # Limit to 20 chunks
            new_num = 20
        
        # Create a new text area
        new_text_area = gr.TextArea(label=f"Chunk {new_num}", placeholder=f"Enter chunk {new_num} of text here...", lines=4)
        
        # Update the remove button to be interactive if we have more than 1 chunk
        remove_btn_update = gr.update(interactive=(new_num > 1))
        
        return new_num, new_text_area, remove_btn_update

    # Function to remove the last text chunk
    def remove_text_chunk(num_chunks):
        new_num = max(1, num_chunks - 1)
        
        # Update the remove button to be interactive if we have more than 1 chunk
        remove_btn_update = gr.update(interactive=(new_num > 1))
        
        return new_num, remove_btn_update

    # Connect add chunk button
    add_chunk_btn.click(
        add_text_chunk,
        inputs=num_chunks,
        outputs=[num_chunks, additional_chunks_container, remove_chunk_btn]
    )

    # Connect remove chunk button
    remove_chunk_btn.click(
        remove_text_chunk,
        inputs=num_chunks,
        outputs=[num_chunks, remove_chunk_btn]
    )

    # Function to collect all text chunks
    def collect_text_chunks(num_chunks, text_chunk_1, *additional_chunks):
        chunks = [text_chunk_1]
        chunks.extend(additional_chunks[:num_chunks-1])  # Only take the actual number of chunks
        return chunks

    # Connect generation button
    gen_button.click(
        generate_text_chunks,
        inputs=[
            prompt_audio,
            num_chunks, text_chunk_1, additional_chunks_container,
            # Emotion control parameters
            emo_control_method, emo_upload, emo_weight,
            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
            emo_text, emo_random,
            # Advanced parameters
            do_sample, top_p, top_k, temperature,
            length_penalty, num_beams, repetition_penalty, max_mel_tokens,
            max_text_tokens_per_segment
        ],
        outputs=[output_zip]
    )

    # Multi-Speaker Mode Tab
    with gr.Tab("Multi Speaker Mode"):
        gr.Markdown("## Multi Speaker Mode\nUpload reference audio for each speaker and create a script with speaker tags.")
        
        with gr.Row():
            num_speakers = gr.Slider(
                label="Number of Speakers",
                minimum=1,
                maximum=10,
                value=2,
                step=1,
                interactive=True
            )
        
        # Create speaker components
        speaker_audios, speaker_names, speaker_rows = create_speaker_ui_components()
        
        # Script input
        script_input = gr.TextArea(
            label="Script",
            placeholder="[SpeakerName]{Text for this speaker}\n[AnotherSpeaker]{Text for another speaker}",
            lines=10,
            info="Use [SpeakerName]{Text} format for each speaker's lines"
        )
        
        # Experimental features checkbox
        experimental_checkbox_multi = gr.Checkbox(label=i18n("显示实验功能"), value=False)
        
        # Function settings accordion
        with gr.Accordion(i18n("功能设置")):
            # Create emotion control components
            (
                emo_control_method_multi, emo_control_method_all_multi,
                emotion_reference_group_multi, emo_upload_multi,
                emotion_randomize_group_multi, emo_random_multi,
                emotion_vector_group_multi, vec1_multi, vec2_multi, vec3_multi, vec4_multi,
                vec5_multi, vec6_multi, vec7_multi, vec8_multi,
                emo_text_group_multi, emo_text_multi,
                emo_weight_group_multi, emo_weight_multi
            ) = create_emotion_control_components_multi()
        
        # Advanced settings accordion
        (
            advanced_settings_group_multi,
            do_sample_multi, top_p_multi, top_k_multi, temperature_multi,
            length_penalty_multi, num_beams_multi, repetition_penalty_multi, max_mel_tokens_multi,
            max_text_tokens_per_segment_multi,
            segments_settings_multi, segments_preview_multi,
            advanced_params_multi
        ) = create_advanced_settings_components_multi()
        
        # Generate button
        generate_multi_btn = gr.Button("Generate All Audio Files", variant="primary")
        
        # Output
        multi_output = gr.File(label="Download Generated Audio (Zip)")
        
        # Error message
        error_message = gr.HTML(visible=False)
        
        # Function to update speaker UI based on number of speakers
        def update_speaker_ui(num_speakers):
            updates = []
            for i in range(len(speaker_rows)):
                if i < num_speakers:
                    updates.append(gr.update(visible=True))
                else:
                    updates.append(gr.update(visible=False))
            return updates
        
        # Update speaker UI when number of speakers changes
        num_speakers.change(
            update_speaker_ui,
            inputs=num_speakers,
            outputs=speaker_rows
        )
        
        # Function to handle emotion control method change in multi-speaker mode
        def on_method_change_multi(emo_control_method):
            if emo_control_method == 1:  # emotion reference audio
                return (gr.update(visible=True),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=True)
                        )
            elif emo_control_method == 2:  # emotion vectors
                return (gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(visible=True),
                        gr.update(visible=False),
                        gr.update(visible=True)
                        )
            elif emo_control_method == 3:  # emotion text description
                return (gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(visible=True)
                        )
            else:  # 0: same as speaker voice
                return (gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False)
                        )
        
        # Connect emotion control method change to UI updates
        emo_control_method_multi.change(
            on_method_change_multi,
            inputs=[emo_control_method_multi],
            outputs=[emotion_reference_group_multi,
                     emotion_randomize_group_multi,
                     emotion_vector_group_multi,
                     emo_text_group_multi,
                     emo_weight_group_multi]
        )
        
        # Function to handle experimental features change in multi-speaker mode
        def on_experimental_change_multi(is_experimental, current_mode_index):
            # 切换情感控制选项
            new_choices = EMO_CHOICES_ALL if is_experimental else EMO_CHOICES_OFFICIAL
            # if their current mode selection doesn't exist in new choices, reset to 0.
            new_index = current_mode_index if current_mode_index < len(new_choices) else 0

            return gr.update(choices=new_choices, value=new_choices[new_index])
        
        # Connect experimental features change to UI updates
        experimental_checkbox_multi.change(
            on_experimental_change_multi,
            inputs=[experimental_checkbox_multi, emo_control_method_multi],
            outputs=[emo_control_method_multi]
        )
        
        # Function to handle script text change in multi-speaker mode
        def on_script_text_change_multi(text, max_text_tokens_per_segment):
            if text and len(text) > 0:
                # Parse the script to extract all text segments
                segments = parse_multi_speaker_script(text)
                all_text = " ".join([segment[1] for segment in segments])
                
                if all_text:
                    text_tokens_list = tts.tokenizer.tokenize(all_text)
                    segments_preview = tts.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=int(max_text_tokens_per_segment))
                    data = []
                    for i, s in enumerate(segments_preview):
                        segment_str = ''.join(s)
                        tokens_count = len(s)
                        data.append([i, segment_str, tokens_count])
                    return {
                        segments_preview_multi: gr.update(value=data, visible=True, type="array"),
                    }
            
            df = pd.DataFrame([], columns=[i18n("序号"), i18n("分句内容"), i18n("Token数")])
            return {
                segments_preview_multi: gr.update(value=df),
            }
        
        # Connect script text changes to preview updates
        script_input.change(
            on_script_text_change_multi,
            inputs=[script_input, max_text_tokens_per_segment_multi],
            outputs=[segments_preview_multi]
        )
        
        max_text_tokens_per_segment_multi.change(
            on_script_text_change_multi,
            inputs=[script_input, max_text_tokens_per_segment_multi],
            outputs=[segments_preview_multi]
        )
        
        # Generate multi-speaker audio
        def generate_multi_speaker_wrapper(num_speakers, script_text, 
                                          # Emotion control parameters
                                          emo_control_method, emo_ref_path, emo_weight,
                                          vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                                          emo_text, emo_random,
                                          # Advanced parameters
                                          do_sample, top_p, top_k, temperature,
                                          length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                                          max_text_tokens_per_segment,
                                          *speaker_components, progress=gr.Progress()):
            try:
                # Extract speaker audios and names from the components
                speaker_audios_list = []
                speaker_names_list = []
                
                # The components are passed as a flat list: [audio1, name1, audio2, name2, ...]
                for i in range(num_speakers):
                    idx = i * 2
                    if idx + 1 < len(speaker_components):
                        speaker_audios_list.append(speaker_components[idx])
                        speaker_names_list.append(speaker_components[idx + 1])
                
                result = generate_multi_speaker_audio(
                    num_speakers, speaker_audios_list, speaker_names_list, script_text,
                    # Emotion control parameters
                    emo_control_method, emo_ref_path, emo_weight,
                    vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                    emo_text, emo_random,
                    # Advanced parameters
                    do_sample, top_p, top_k, temperature,
                    length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                    max_text_tokens_per_segment,
                    progress
                )
                if result is None:
                    return None, gr.update(visible=True, value="<div style='color: red; font-weight: bold;'>Error: Failed to generate audio. Please check your inputs and try again.</div>")
                return result, gr.update(visible=False)
            except Exception as e:
                error_msg = f"<div style='color: red; font-weight: bold;'>Error: {str(e)}</div>"
                return None, gr.update(visible=True, value=error_msg)
        
        # Flatten the speaker components for the function input
        speaker_components_flat = []
        for audio, name in zip(speaker_audios, speaker_names):
            speaker_components_flat.append(audio)
            speaker_components_flat.append(name)
        
        generate_multi_btn.click(
            generate_multi_speaker_wrapper,
            inputs=[
                num_speakers, script_input,
                # Emotion control parameters
                emo_control_method_multi, emo_upload_multi, emo_weight_multi,
                vec1_multi, vec2_multi, vec3_multi, vec4_multi,
                vec5_multi, vec6_multi, vec7_multi, vec8_multi,
                emo_text_multi, emo_random_multi,
                # Advanced parameters
                do_sample_multi, top_p_multi, top_k_multi, temperature_multi,
                length_penalty_multi, num_beams_multi, repetition_penalty_multi, max_mel_tokens_multi,
                max_text_tokens_per_segment_multi
            ] + speaker_components_flat,
            outputs=[multi_output, error_message]
        )

if __name__ == "__main__":
    try:
        demo.queue(20)
        demo.launch(server_name=cmd_args.host, server_port=cmd_args.port, share=True, debug=True)
    except Exception as e:
        print(f"Error launching web UI: {e}")
        traceback.print_exc()
