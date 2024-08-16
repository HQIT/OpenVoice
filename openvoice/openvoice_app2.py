import os
import torch
import argparse
import gradio as gr
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--share", action='store_true', default=False, help="make link public")
args = parser.parse_args()

ckpt_converter = 'checkpoints_v2/converter'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = 'outputs_v2'
os.makedirs(output_dir, exist_ok=True)

# load models
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

language_ttsmodels_cache = {
    'ZH':None,
    'EN':None,
    'EN_NEWEST':None,
    'ES':None,
    'FR':None,
    'JP':None,
    'KR':None
}

# Cache for language and speakers
language_speakers_cache = {
    'ZH':[],
    'EN':[],
    'EN_NEWEST':[],
    'ES':[],
    'FR':[],
    'JP':[],
    'KR':[]
}

for language, _ in language_ttsmodels_cache.items():
    model = TTS(language=language, device=device)
    speaker_ids = model.hps.data.spk2id
    
    for speaker_key in speaker_ids.keys():
        speaker_id = speaker_ids[speaker_key]
        speaker_key = speaker_key.lower().replace('_', '-')
        language_speakers_cache[language].append(speaker_id)
        
        source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
        language_ttsmodels_cache[language] = (model, source_se)

def update_speakers(language):
    return gr.Dropdown(choices=language_speakers_cache[language], value=language_speakers_cache[language][0])

def predict(prompt, language, speaker, audio_file_pth, agree):
    # initialize a empty info
    text_hint = ''
    # agree with the terms
    if agree == False:
        text_hint += '[ERROR] Please accept the Terms & Condition!\n'
        gr.Warning("Please accept the Terms & Condition!")
        return (
            text_hint,
            None,
            None,
        )

    speaker_wav = audio_file_pth

    if len(prompt) < 2:
        text_hint += f"[ERROR] Please give a longer prompt text \n"
        gr.Warning("Please give a longer prompt text")
        return (
            text_hint,
            None,
            None,
        )
    if len(prompt) > 200:
        text_hint += f"[ERROR] Text length limited to 200 characters for this demo, please try shorter text. You can clone our open-source repo and try for your usage \n"
        gr.Warning(
            "Text length limited to 200 characters for this demo, please try shorter text. You can clone our open-source repo for your usage"
        )
        return (
            text_hint,
            None,
            None,
        )
    
    print(f"!!! speaker_wav: {speaker_wav}")

    target_se, audio_name = None, None
    try:
        if speaker_wav:
            target_se, audio_name = se_extractor.get_se(speaker_wav, tone_color_converter, vad=False)
    except Exception as e:
        text_hint += f"[ERROR] Get target tone color error {str(e)} \n"
        gr.Warning(
            "[ERROR] Get target tone color error {str(e)} \n"
        )
        return (
            text_hint,
            None,
            None,
        )

    src_path = f'{output_dir}/tmp.wav'
    tts_model, source_se = language_ttsmodels_cache[language]
    speaker_id = speaker

    print(f"!!! speaker_id: {speaker_id}, language: {language}, speaker: {speaker}, src_path: {src_path}, prompt: {prompt}")
    tts_model.tts_to_file(prompt, speaker_id, src_path, speed=1.0)

    save_path = f'{output_dir}/output.wav'

    if target_se is not None and target_se.numel() > 0:
        # Run the tone color converter
        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=src_path, 
            src_se=source_se, 
            tgt_se=target_se, 
            output_path=save_path,
            message=encode_message)
    else:
        shutil.copy(src_path, save_path)

    text_hint += f'''Get response successfully \n'''

    return (
        text_hint,
        save_path,
        speaker_wav,
    )

title = "MyShell OpenVoice"

with gr.Blocks(analytics_enabled=False) as demo:
    with gr.Row():
        with gr.Column():
            input_text_gr = gr.Textbox(
                label="Text Prompt",
                info="One or two sentences at a time is better. Up to 200 text characters.",
                value="He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered, flour-fattened sauce.",
            )
            language_gr = gr.Radio(
                label="Language",
                choices=list(language_speakers_cache.keys()),
                value='ZH'
            )
            speaker_gr = gr.Dropdown(
                label="Speaker",
                info="",
                choices=language_speakers_cache['ZH'],
                value=language_speakers_cache['ZH'][0] if language_speakers_cache['ZH'] else 'Unknown'
            )
            ref_gr = gr.Audio(
                label="Reference Audio",
                info="Click on the ✎ button to upload your own target speaker audio",
                type="filepath",
                value="resources/demo_speaker2.mp3",
            )
            tos_gr = gr.Checkbox(
                label="Agree",
                value=True,
                info="I agree to the terms of the cc-by-nc-4.0 license-: https://github.com/myshell-ai/OpenVoice/blob/main/LICENSE",
            )

            tts_button = gr.Button("Send", elem_id="send-btn", visible=True)

        with gr.Column():
            out_text_gr = gr.Text(label="Info")
            audio_gr = gr.Audio(label="Synthesised Audio", autoplay=True)
            ref_audio_gr = gr.Audio(label="Reference Audio Used")
            tts_button.click(predict, [input_text_gr, language_gr, speaker_gr, ref_gr, tos_gr], outputs=[out_text_gr, audio_gr, ref_audio_gr])
    
    language_gr.change(fn=update_speakers, inputs=language_gr, outputs=speaker_gr)
demo.queue()  
demo.launch(debug=True, show_api=True, share=args.share)
