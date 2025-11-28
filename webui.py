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
import argparse

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import torch
import gradio as gr

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

# --- Argument Parsing ---
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

# --- Validation ---
if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in ["bpe.model", "gpt.pth", "config.yaml", "s2mel.pth", "wav2vec2bert_stats.pt"]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

# --- Import Model ---
# Assuming indextts is available in the path
try:
    from indextts.infer_v2 import IndexTTS2
    from tools.i18n.i18n import I18nAuto
except ImportError as e:
    print(f"Failed to import IndexTTS modules: {e}")
    sys.exit(1)

i18n = I18nAuto(language="Auto")

# --- Model Initialization ---
def initialize_tts_model():
    try:
        print("Initializing IndexTTS2 model...")
        tts_instance = IndexTTS2(
            model_dir=cmd_args.model_dir,
            cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
            use_fp16=cmd_args.fp16,
            use_deepspeed=cmd_args.deepspeed,
            use_cuda_kernel=cmd_args.cuda_kernel,
        )
        print("Model initialized successfully!")
        return tts_instance
    except Exception as e:
        print(f"Error initializing model: {e}")
        traceback.print_exc()
        return None

tts = initialize_tts_model()
if tts is None:
    print("Failed to initialize TTS model. Exiting.")
    sys.exit(1)

# --- Constants & Helpers ---
LANGUAGES = {
    "中文": "zh_CN",
    "English": "en_US"
}
EMO_CHOICES_ALL = [
    i18n("与音色参考音频相同"),
    i18n("使用情感参考音频"),
    i18n("使用情感向量控制"),
    i18n("使用情感描述文本控制")
]
EMO_CHOICES_OFFICIAL = EMO_CHOICES_ALL[:-1]  # skip experimental features

os.makedirs("outputs/tasks", exist_ok=True)
os.makedirs("prompts", exist_ok=True)

example_cases = []
if os.path.exists("examples/cases.jsonl"):
    with open("examples/cases.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            example = json.loads(line)
            emo_audio_path = os.path.join("examples", example["emo_audio"]) if example.get("emo_audio") else None
            
            example_cases.append([
                os.path.join("examples", example.get("prompt_audio", "sample_prompt.wav")),
                EMO_CHOICES_ALL[example.get("emo_mode", 0)],
                example.get("text"),
                emo_audio_path,
                example.get("emo_weight", 1.0),
                example.get("emo_text", ""),
                example.get("emo_vec_1", 0), example.get("emo_vec_2", 0),
                example.get("emo_vec_3", 0), example.get("emo_vec_4", 0),
                example.get("emo_vec_5", 0), example.get("emo_vec_6", 0),
                example.get("emo_vec_7", 0), example.get("emo_vec_8", 0),
            ])

def get_example_cases(include_experimental=False):
    if include_experimental:
        return example_cases
    return [x for x in example_cases if x[1] != EMO_CHOICES_ALL[3]]

def create_warning_message(warning_text):
    return gr.HTML(f"<div style=\"padding: 0.5em 0.8em; border-radius: 0.5em; background: #ffa87d; color: #000; font-weight: bold\">{html.escape(warning_text)}</div>")

def create_experimental_warning_message():
    return create_warning_message(i18n('提示：此功能为实验版，结果尚不稳定，我们正在持续优化中。'))

# --- Logic Functions ---

def parse_multi_speaker_script(script_text):
    pattern = r'\[([^\]]+)\]\{([^}]*)\}'
    matches = re.findall(pattern, script_text)
    return [(speaker, text.strip()) for speaker, text in matches]

def generate_text_chunks_logic(prompt, num_chunks, text_chunks_list, 
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
        # Collect valid chunks based on num_chunks
        valid_texts = []
        for i in range(min(len(text_chunks_list), int(num_chunks))):
            txt = text_chunks_list[i]
            if txt and txt.strip():
                valid_texts.append((i, txt))
        
        if not valid_texts:
            raise ValueError("No valid text chunks found. Please enter some text.")
        
        temp_dir = tempfile.mkdtemp()
        generated_files = []
        
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
            # Handle radio button value (string vs index)
            if emo_control_method in EMO_CHOICES_ALL:
                emo_control_method = EMO_CHOICES_ALL.index(emo_control_method)
            else:
                emo_control_method = 0 # Fallback

        if emo_control_method == 0: 
            emo_ref_path = None
        elif emo_control_method == 2:
            vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
            vec = tts.normalize_emo_vec(vec, apply_bias=True)
        else:
            vec = None

        if not emo_text: emo_text = None
        
        total_chunks = len(valid_texts)
        for idx, (original_idx, text) in enumerate(valid_texts):
            progress(idx / total_chunks, desc=f"Generating chunk {idx+1}/{total_chunks}")
            output_path = os.path.join(temp_dir, f"chunk_{original_idx+1}.wav")
            
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
        
        zip_path = os.path.join(temp_dir, "text_chunks_audios.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in generated_files:
                zipf.write(file, os.path.basename(file))
        
        return zip_path
    except Exception as e:
        print(f"Error in generate_text_chunks: {e}")
        traceback.print_exc()
        return None

def generate_multi_speaker_audio(num_speakers, speaker_audios, speaker_names, script_text,
                                emo_control_method, emo_ref_path, emo_weight,
                                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                                emo_text, emo_random,
                                do_sample, top_p, top_k, temperature,
                                length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                                max_text_tokens_per_segment,
                                progress=gr.Progress()):
    try:
        speaker_map = {}
        for i in range(num_speakers):
            name = speaker_names[i]
            audio = speaker_audios[i]
            if name and audio:
                speaker_map[name] = audio
        
        segments = parse_multi_speaker_script(script_text)
        if not segments:
            raise ValueError("No valid script segments found. Please use [SpeakerName]{Text} format.")
        
        temp_dir = tempfile.mkdtemp()
        generated_files = []
        
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
        
        # Determine emotion mode index
        if type(emo_control_method) is not int:
             if emo_control_method in EMO_CHOICES_ALL:
                emo_control_method = EMO_CHOICES_ALL.index(emo_control_method)
             else:
                emo_control_method = 0

        if emo_control_method == 0: emo_ref_path = None
        
        if emo_control_method == 2:
            vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
            vec = tts.normalize_emo_vec(vec, apply_bias=True)
        else:
            vec = None

        if not emo_text: emo_text = None
        
        total_segments = len(segments)
        for i, (speaker, text) in enumerate(segments):
            progress(i / total_segments, desc=f"Generating segment {i+1}/{total_segments} for speaker: {speaker}")
            
            if speaker not in speaker_map:
                print(f"Warning: Speaker '{speaker}' not found in speaker list. Skipping.")
                continue
                
            output_path = os.path.join(temp_dir, f"{speaker}_{i}.wav")
            
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
            raise ValueError("No audio files were generated.")
        
        zip_path = os.path.join(temp_dir, "multi_speaker_audios.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in generated_files:
                zipf.write(file, os.path.basename(file))
        
        return zip_path
    except Exception as e:
        print(f"Error in generate_multi_speaker_audio: {e}")
        traceback.print_exc()
        return None

# --- UI Component Creators ---

def create_speaker_ui_components(max_speakers=10):
    speaker_audios = []
    speaker_names = []
    speaker_rows = []
    
    for i in range(max_speakers):
        with gr.Row(visible=(i < 2)) as speaker_row:
            speaker_audio = gr.Audio(label=f"Speaker {i+1} Reference Audio", type="filepath")
            speaker_name = gr.Textbox(label=f"Speaker {i+1} Name", placeholder=f"Enter name for speaker {i+1}")
            speaker_audios.append(speaker_audio)
            speaker_names.append(speaker_name)
            speaker_rows.append(speaker_row)
    
    return speaker_audios, speaker_names, speaker_rows

def create_emotion_control_components(prefix=""):
    # Shared function to create emotion UI elements to avoid code duplication
    choices_comp = gr.Radio(
        choices=EMO_CHOICES_OFFICIAL,
        type="index",
        value=EMO_CHOICES_OFFICIAL[0],
        label=i18n("情感控制方式")
    )
    choices_all_comp = gr.Radio(
        choices=EMO_CHOICES_ALL,
        type="index",
        value=EMO_CHOICES_ALL[0],
        label=i18n("情感控制方式"),
        visible=False
    )
    
    with gr.Group(visible=False) as ref_group:
        with gr.Row():
            upload_comp = gr.Audio(label=i18n("上传情感参考音频"), type="filepath")
    
    with gr.Row(visible=False) as random_group:
        random_comp = gr.Checkbox(label=i18n("情感随机采样"), value=False)
    
    with gr.Group(visible=False) as vec_group:
        with gr.Row():
            with gr.Column():
                v1 = gr.Slider(label=i18n("喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                v2 = gr.Slider(label=i18n("怒"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                v3 = gr.Slider(label=i18n("哀"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                v4 = gr.Slider(label=i18n("惧"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
            with gr.Column():
                v5 = gr.Slider(label=i18n("厌恶"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                v6 = gr.Slider(label=i18n("低落"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                v7 = gr.Slider(label=i18n("惊喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                v8 = gr.Slider(label=i18n("平静"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
    
    with gr.Group(visible=False) as text_group:
        create_experimental_warning_message()
        with gr.Row():
            text_comp = gr.Textbox(
                label=i18n("情感描述文本"),
                placeholder=i18n("请输入情绪描述"),
                value="",
                info=i18n("例如：委屈巴巴、危险在悄悄逼近")
            )
            
    with gr.Row(visible=False) as weight_group:
        weight_comp = gr.Slider(label=i18n("情感权重"), minimum=0.0, maximum=1.0, value=0.65, step=0.01)

    return (choices_comp, choices_all_comp, ref_group, upload_comp, random_group, random_comp,
            vec_group, v1, v2, v3, v4, v5, v6, v7, v8, text_group, text_comp, weight_group, weight_comp)

def create_advanced_settings_components():
    with gr.Accordion(i18n("高级生成参数设置"), open=False, visible=True) as adv_group:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(f"**{i18n('GPT2 采样设置')}**")
                with gr.Row():
                    do_sample = gr.Checkbox(label="do_sample", value=True, info=i18n("是否进行采样"))
                    temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                with gr.Row():
                    top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                    top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                    num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                with gr.Row():
                    repetition_penalty = gr.Number(label="repetition_penalty", value=10.0, minimum=0.1, maximum=20.0)
                    length_penalty = gr.Number(label="length_penalty", value=0.0, minimum=-2.0, maximum=2.0)
                max_mel_tokens = gr.Slider(label="max_mel_tokens", value=1500, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10)
            with gr.Column(scale=2):
                gr.Markdown(f'**{i18n("分句设置")}**')
                with gr.Row():
                    initial_val = max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                    max_text_tokens = gr.Slider(label=i18n("分句最大Token数"), value=initial_val, minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2)
                with gr.Accordion(i18n("预览分句结果"), open=True):
                    seg_preview = gr.Dataframe(headers=[i18n("序号"), i18n("分句内容"), i18n("Token数")], wrap=True)
    
    return (adv_group, do_sample, top_p, top_k, temperature, length_penalty, num_beams, 
            repetition_penalty, max_mel_tokens, max_text_tokens, seg_preview)


# --- Application Construction ---

with gr.Blocks(title="IndexTTS Demo") as demo:
    gr.HTML('''
    <h2><center>IndexTTS2: Emotionally Expressive Zero-Shot Text-to-Speech</center></h2>
    ''')

    # === Tab 1: Single Speaker / Chunks ===
    with gr.Tab(i18n("音频生成")):
        with gr.Row():
            prompt_audio = gr.Audio(label=i18n("音色参考音频"), sources=["upload","microphone"], type="filepath")
            
            with gr.Column():
                gr.Markdown("### Text Chunks")
                gr.Markdown("Enter your text below. Use 'Add/Remove' to manage chunks.")
                
                # PRE-CREATE chunks for dynamic UI
                MAX_CHUNKS = 20
                chunk_components = []
                with gr.Column():
                    for i in range(MAX_CHUNKS):
                        # Only show first chunk by default
                        vis = True if i == 0 else False
                        t = gr.TextArea(label=f"Chunk {i+1}", visible=vis, lines=3)
                        chunk_components.append(t)
                
                num_chunks_state = gr.State(value=1)
                
                with gr.Row():
                    add_chunk_btn = gr.Button("Add Chunk")
                    remove_chunk_btn = gr.Button("Remove Last Chunk", interactive=False)
                
                gen_button = gr.Button(i18n("生成语音"), variant="primary")
                output_zip = gr.File(label="Download Audio (Zip)", visible=True)

        experimental_checkbox = gr.Checkbox(label=i18n("显示实验功能"), value=False)

        with gr.Accordion(i18n("功能设置")):
            (emo_method, emo_method_all, emo_ref_grp, emo_up, emo_rnd_grp, emo_rnd,
             emo_vec_grp, v1, v2, v3, v4, v5, v6, v7, v8, 
             emo_txt_grp, emo_txt, emo_w_grp, emo_w) = create_emotion_control_components()

        (adv_grp, do_sample, top_p, top_k, temp, len_pen, beams, rep_pen, max_mel, max_tok, seg_prev) = create_advanced_settings_components()

        # Logic for Chunks UI
        def update_chunks_visibility(count):
            updates = []
            for i in range(MAX_CHUNKS):
                updates.append(gr.update(visible=(i < count)))
            # Update remove button interactivity
            remove_interactive = gr.update(interactive=(count > 1))
            return updates + [remove_interactive]

        def add_chunk_fn(count):
            new_count = min(count + 1, MAX_CHUNKS)
            return [new_count] + update_chunks_visibility(new_count)

        def remove_chunk_fn(count):
            new_count = max(count - 1, 1)
            return [new_count] + update_chunks_visibility(new_count)

        add_chunk_btn.click(add_chunk_fn, inputs=[num_chunks_state], outputs=[num_chunks_state] + chunk_components + [remove_chunk_btn])
        remove_chunk_btn.click(remove_chunk_fn, inputs=[num_chunks_state], outputs=[num_chunks_state] + chunk_components + [remove_chunk_btn])

        # Wrapper to handle flattened inputs
        def generate_chunks_wrapper(prompt, num_chunks, *args):
            # The args contain: [20 chunk texts, emotion params..., advanced params...]
            text_values = args[:MAX_CHUNKS]
            remaining_args = args[MAX_CHUNKS:]
            
            return generate_text_chunks_logic(
                prompt, num_chunks, text_values, 
                *remaining_args
            )

        gen_button.click(
            generate_chunks_wrapper,
            inputs=[prompt_audio, num_chunks_state] + chunk_components + [
                emo_method, emo_up, emo_w,
                v1, v2, v3, v4, v5, v6, v7, v8,
                emo_txt, emo_rnd,
                do_sample, top_p, top_k, temp, len_pen, beams, rep_pen, max_mel, max_tok
            ],
            outputs=[output_zip]
        )
        
        # Example Handler
        example_table = gr.Dataset(
            label="Examples",
            samples=get_example_cases(False),
            components=[prompt_audio, emo_method_all, chunk_components[0], emo_up, emo_w, emo_txt, v1, v2, v3, v4, v5, v6, v7, v8],
            type="values"
        )
        
        def on_example_click(ex):
            return (
                gr.update(value=ex[0]), # prompt
                gr.update(value=ex[1]), # method
                gr.update(value=ex[2]), # text (chunk 1)
                gr.update(value=ex[3]), # emo audio
                gr.update(value=ex[4]), # weight
                gr.update(value=ex[5]), # emo text
                gr.update(value=ex[6]), gr.update(value=ex[7]), gr.update(value=ex[8]), gr.update(value=ex[9]),
                gr.update(value=ex[10]), gr.update(value=ex[11]), gr.update(value=ex[12]), gr.update(value=ex[13])
            )
        
        example_table.click(on_example_click, inputs=[example_table], 
                            outputs=[prompt_audio, emo_method, chunk_components[0], emo_up, emo_w, emo_txt, v1, v2, v3, v4, v5, v6, v7, v8])


    # === Tab 2: Multi-Speaker ===
    with gr.Tab("Multi Speaker Mode"):
        gr.Markdown("## Multi Speaker Mode")
        
        num_speakers_slider = gr.Slider(label="Number of Speakers", minimum=1, maximum=10, value=2, step=1)
        spk_audios, spk_names, spk_rows = create_speaker_ui_components()
        
        script_input = gr.TextArea(label="Script", placeholder="[SpeakerName]{Text}", lines=10)
        
        experimental_checkbox_multi = gr.Checkbox(label=i18n("显示实验功能"), value=False)
        
        with gr.Accordion(i18n("功能设置")):
            (m_emo_method, m_emo_method_all, m_emo_ref_grp, m_emo_up, m_emo_rnd_grp, m_emo_rnd,
             m_emo_vec_grp, mv1, mv2, mv3, mv4, mv5, mv6, mv7, mv8, 
             m_emo_txt_grp, m_emo_txt, m_emo_w_grp, m_emo_w) = create_emotion_control_components()

        (m_adv_grp, m_do_sample, m_top_p, m_top_k, m_temp, m_len_pen, m_beams, m_rep_pen, m_max_mel, m_max_tok, m_seg_prev) = create_advanced_settings_components()

        multi_gen_btn = gr.Button("Generate All Audio Files", variant="primary")
        multi_output = gr.File(label="Download Generated Audio (Zip)")
        error_box = gr.HTML(visible=False)

        # Logic for Speaker UI Updates
        def update_speaker_vis(count):
            updates = []
            for i in range(len(spk_rows)):
                updates.append(gr.update(visible=(i < count)))
            return updates

        num_speakers_slider.change(update_speaker_vis, inputs=num_speakers_slider, outputs=spk_rows)

        # Logic for Multi-Gen Wrapper
        def multi_gen_wrapper(num_spk, script, *args):
            # Args structure: 
            # Emotion params...
            # Advanced params...
            # Speaker Audios (10)
            # Speaker Names (10)
            
            # Helper to calculate offsets based on component counts in list
            # Fixed params count:
            n_emo = 16 # method, up, weight, 8 vecs, text, random (check inputs below)
            n_adv = 9
            
            # Since we pass specific inputs, let's index manually based on the input list below
            emo_args = args[0:16]
            adv_args = args[16:25]
            spk_args = args[25:]
            
            # Separate audios and names
            # inputs passed: spk_audios + spk_names
            current_audios = spk_args[0:10]
            current_names = spk_args[10:20]
            
            res = generate_multi_speaker_audio(
                num_spk, current_audios, current_names, script,
                *emo_args, *adv_args
            )
            
            if res is None:
                return None, gr.update(visible=True, value="<div style='color:red'>Generation Failed</div>")
            return res, gr.update(visible=False)

        multi_gen_btn.click(
            multi_gen_wrapper,
            inputs=[num_speakers_slider, script_input,
                    m_emo_method, m_emo_ref_grp, m_emo_w, # Note: using group as placeholder for path? No, need actual upload comp
                    mv1, mv2, mv3, mv4, mv5, mv6, mv7, mv8,
                    m_emo_txt, m_emo_rnd,
                    # Correction: Inputs must match function args strictly
                    # Re-mapping inputs for clarity:
                    m_emo_method, m_emo_up, m_emo_w,
                    mv1, mv2, mv3, mv4, mv5, mv6, mv7, mv8,
                    m_emo_txt, m_emo_rnd,
                    m_do_sample, m_top_p, m_top_k, m_temp, m_len_pen, m_beams, m_rep_pen, m_max_mel, m_max_tok
                   ] + spk_audios + spk_names,
            outputs=[multi_output, error_box]
        )

    # === Shared Logic (Visibility Toggles) ===
    def on_method_change(method):
        # 0:Speaker, 1:AudioRef, 2:Vector, 3:Text
        if type(method) is not int: return [gr.update()]*5 # basic safety
        
        vis_ref = (method == 1)
        vis_vec = (method == 2)
        vis_txt = (method == 3)
        vis_w = (method != 0)
        
        return (
            gr.update(visible=vis_ref), # ref group
            gr.update(visible=vis_vec), # random group (usually with vector)
            gr.update(visible=vis_vec), # vector group
            gr.update(visible=vis_txt), # text group
            gr.update(visible=vis_w)    # weight group
        )

    # Link Visibility - Single
    emo_method.change(on_method_change, inputs=[emo_method], 
                      outputs=[emo_ref_grp, emo_rnd_grp, emo_vec_grp, emo_txt_grp, emo_w_grp])
    
    # Link Visibility - Multi
    m_emo_method.change(on_method_change, inputs=[m_emo_method], 
                        outputs=[m_emo_ref_grp, m_emo_rnd_grp, m_emo_vec_grp, m_emo_txt_grp, m_emo_w_grp])
    
    # Experimental toggles
    def on_exp_change(is_exp, current):
        choices = EMO_CHOICES_ALL if is_exp else EMO_CHOICES_OFFICIAL
        val = current if current < len(choices) else 0
        return gr.update(choices=choices, value=val)

    experimental_checkbox.change(on_exp_change, inputs=[experimental_checkbox, emo_method], outputs=[emo_method])
    experimental_checkbox_multi.change(on_exp_change, inputs=[experimental_checkbox_multi, m_emo_method], outputs=[m_emo_method])
    
    # Text Token Analysis (Preview)
    def on_text_change(text, max_tok):
        if not text: return gr.update(value=[])
        tokens = tts.tokenizer.tokenize(text)
        segs = tts.tokenizer.split_segments(tokens, int(max_tok))
        data = [[i, ''.join(s), len(s)] for i,s in enumerate(segs)]
        return gr.update(value=data)

    # Use first chunk for preview in single mode
    chunk_components[0].change(on_text_change, inputs=[chunk_components[0], max_tok], outputs=[seg_prev])
    
    # Script parsing for multi mode
    def on_script_change(text, max_tok):
        if not text: return gr.update(value=[])
        pairs = parse_multi_speaker_script(text)
        full_text = " ".join([p[1] for p in pairs])
        return on_text_change(full_text, max_tok)
        
    script_input.change(on_script_change, inputs=[script_input, m_max_tok], outputs=[m_seg_prev])


if __name__ == "__main__":
    try:
        demo.queue(20)
        demo.launch(server_name=cmd_args.host, server_port=cmd_args.port, share=True) # debug=True removed for prod
    except Exception as e:
        print(f"Error launching web UI: {e}")
        traceback.print_exc()
