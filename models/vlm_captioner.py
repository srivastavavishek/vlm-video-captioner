from pathlib import Path
import av
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration


class VLMCaptioner:
    """
    VLM Engine wrapper powered by Qwen2-VL.
    """
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    ):
        self.model_id = model_id
        self.processor = None
        self.model = None

    def load_model(self) -> None:
        """
        Loads weights into memory lazily.
        """
        if self.model is not None and self.processor is not None:
            return

        try:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                quantization_config=quant_config,
                device_map="auto",
                low_cpu_mem_usage=True
            )

            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model.eval()

        except Exception as e:
            self.model = None
            self.processor = None
            raise e

    def _extract_frames(
        self,
        video_path: str,
        num_frames: int = 6,
    ) -> list[Image.Image]:
        """
        Extract evenly spaced frames without storing the entire video in memory.
        """
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video chunk not found: {video_path}")

        if num_frames <= 0:
            raise ValueError("num_frames must be greater than 0")

        frames = []

        try:
            with av.open(str(video_path)) as container:
                stream = container.streams.best("video")
                if stream is None:
                    raise ValueError(f"No video stream found in {video_path}")

                total_frames = stream.frames

                if total_frames <= 0:
                    duration = (
                        float(stream.duration * stream.time_base)
                        if stream.duration
                        else float(container.duration / av.time_base)
                    )

                    fps = (
                        float(stream.average_rate)
                        if stream.average_rate
                        and float(stream.average_rate) > 0
                        else 30.0
                    )

                    total_frames = max(1, int(duration * fps))

                # Select evenly spaced frame indices.
                target_indices = set(
                    np.linspace(
                        0,
                        total_frames - 1,
                        min(num_frames, total_frames),
                        dtype=int,
                    )
                )

                for idx, frame in enumerate(container.decode(stream)):
                    if idx in target_indices:
                        frames.append(frame.to_image())

                    if len(frames) == len(target_indices):
                        break

        except Exception as exc:
            print(f"Frame extraction error for {video_path}: {exc}")

        return frames

    @torch.inference_mode()
    def caption_chunk(
        self,
        video_path: str,
        prompt: str,
        num_frames: int = 6
    ) -> str:
        """
        Caption a video chunk using Qwen2-VL.
        """
        self.load_model()
        device = next(self.model.parameters()).device

        frames = self._extract_frames(video_path, num_frames=num_frames)

        if not frames:
            return ""

        content = [
            {"type": "image", "image": img}
            for img in frames
        ]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_prompt],
            images=frames,
            padding=True,
            return_tensors="pt"
        ).to(device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=1,
            do_sample=False
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        del inputs, generated_ids, generated_ids_trimmed
        return output_text[0].strip() if output_text else ""

    def caption_chunks(
        self,
        chunk_paths: list[str],
        prompt: str,
        num_frames: int = 6
    ) -> list[str]:
        """
        Caption video chunks in sequence.
        """
        self.load_model()

        results = []

        for chunk_path in chunk_paths:
            try:
                caption = self.caption_chunk(
                    chunk_path,
                    prompt=prompt,
                    num_frames=num_frames,
                )
                results.append(caption)
            except Exception as e:
                print(f"Error processing {chunk_path}: {e}")
                results.append("")

        return results

    def offload(self) -> None:
        """Clears intermediate CUDA caches."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
