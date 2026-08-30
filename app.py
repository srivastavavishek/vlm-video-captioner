import shutil
from pathlib import Path
import gradio as gr

from utils.video_chunker import VideoChunker
from models.vlm_captioner import VLMCaptioner
from models.nllb_translator import NLLBTranslator
from utils.subtitle_generator import SubtitleGenerator
from utils.language_codes import LANGUAGE_CONFIG
from utils.prompt import prompt


def process_video_pipeline(video_input: str, language_input: str):
    # Validate input and clear previous state
    if not video_input:
        yield (
            "STATUS: ERROR - PLEASE UPLOAD A VIDEO FIRST.",
            gr.update(value=None, subtitles=None),
            "",
            None,
            None
        )
        return

    # Ensure persistent output directory exists in Colab environment
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy input video to outputs directory to prevent temporary file deletion issues
    output_video_filename = Path(video_input).name
    persistent_video_path = output_dir / output_video_filename
    shutil.copy(str(video_input), str(persistent_video_path))
    video_path = str(persistent_video_path)

    selected_language = language_input
    lang_info = LANGUAGE_CONFIG.get(selected_language, LANGUAGE_CONFIG["English"])
    target_flores_code = lang_info["code"]
    target_filename = lang_info["filename"]

    # Step 1: Video Chunking
    yield "STATUS: SPLITTING VIDEO INTO CHUNKS...", gr.update(value=video_path), "", None, None

    try:
        chunker = VideoChunker(output_dir="temp_chunks")
        chunks_metadata = chunker.split_video(video_path, chunk_duration=5)
        chunk_paths = [meta["chunk_path"] for meta in chunks_metadata]
    except Exception as exc:
        yield f"STATUS: ERROR DURING CHUNKING - {exc}", gr.update(value=video_path), "", None, None
        return

    # Step 2: Extract Frames & Generate English Captions (VLM)
    yield "STATUS: EXTRACTING FRAMES & GENERATING CAPTIONS (VLM)...", gr.update(value=video_path), "", None, None

    captioner = VLMCaptioner()

    try:
        english_captions = captioner.caption_chunks(
            chunk_paths=chunk_paths,
            prompt=prompt,
            num_frames=7
        )
    except Exception as exc:
        yield f"STATUS: ERROR DURING VLM CAPTIONING - {exc}", gr.update(value=video_path), "", None, None
        return
    finally:
        captioner.offload()

    # Step 3: Translate Captions (NLLB)
    yield f"STATUS: TRANSLATING CAPTIONS TO {selected_language.upper()}...", gr.update(value=video_path), "", None, None

    translator = NLLBTranslator()
    try:
        translated_captions = translator.translate_batch(
            captions=english_captions,
            src_lang="eng_Latn",
            tgt_lang=target_flores_code,
            batch_size=16
        )
    except Exception as exc:
        yield f"STATUS: ERROR DURING TRANSLATION - {exc}", gr.update(value=video_path), "", None, None
        return
    finally:
        translator.offload_from_vram()

    # Step 4: Generate WebVTT & SRT Subtitles
    yield "STATUS: PREPARING SUBTITLES...", gr.update(value=video_path), "", None, None

    sub_gen = SubtitleGenerator()
    english_payload = sub_gen.build_payload(chunks_metadata, english_captions)
    translated_payload = sub_gen.build_payload(chunks_metadata, translated_captions)

    subtitle_vtt_path = f"outputs/{target_filename}.vtt"
    english_srt_path = "outputs/english.srt"
    target_srt_path = f"outputs/{target_filename}.srt"

    sub_gen.export_vtt(translated_payload, subtitle_vtt_path)
    sub_gen.export_srt(english_payload, english_srt_path)
    sub_gen.export_srt(translated_payload, target_srt_path)

    # Step 5: Render Native Player & Return Final Results
    with open(subtitle_vtt_path, "r", encoding="utf-8") as f:
        caption_preview_text = f.read()

    yield (
        "STATUS: DONE",
        # Dynamically inject the output subtitle file into the video player
        gr.update(value=video_path, subtitles=target_srt_path),
        caption_preview_text,
        english_srt_path,
        target_srt_path
    )

with gr.Blocks(title="Video Captioning & Subtitle System") as demo:
    gr.Markdown("# Automated Multilingual Video Subtitle Generator")

    with gr.Row():
        video_component = gr.Video(label="Upload/Drop Video")
        dropdown_component = gr.Dropdown(
            choices=list(LANGUAGE_CONFIG.keys()),
            value="English",
            label="Target Subtitle Language"
        )

    generate_btn = gr.Button("Generate Subtitles", variant="primary")

    status_output = gr.Textbox(
        label="Status",
        value="STATUS: AWAITING INPUT...",
        interactive=False
    )

    # Swap gr.HTML for gr.Video to utilize native Gradio subtitle rendering
    player_output = gr.Video(label="Video Player with Subtitles", interactive=False, subtitles=None)

    preview_output = gr.Textbox(
        label="Caption Preview (WebVTT Source)",
        lines=6,
        interactive=False
    )

    with gr.Row():
        dl_english = gr.File(label="Download Base English SRT")
        dl_target = gr.File(label="Download Target Language SRT")

    generate_btn.click(
        fn=process_video_pipeline,
        inputs=[video_component, dropdown_component],
        outputs=[
            status_output,
            player_output,
            preview_output,
            dl_english,
            dl_target
        ]
    )
