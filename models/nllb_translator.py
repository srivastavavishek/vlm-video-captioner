import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

class NLLBTranslator:
    """
    Sequence-to-Sequence Translation Engine powered by NLLB-200.
    """

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M"
    ):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load_model(self) -> None:
        """Loads NLLB-200 weights and tokenizer lazily."""
        if self.model is not None and self.tokenizer is not None:
            return

        try:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model.eval()

        except Exception as e:
            self.model = None
            self.tokenizer = None
            raise e

    def _deduplicate(self, texts: list[str]) -> tuple[list[str], list[int]]:
        """Stores each unique caption once and returns index mapping."""
        unique_texts = []
        text_to_index = {}
        index_map = []

        for text in texts:
            cleaned_text = text.strip()
            if cleaned_text not in text_to_index:
                text_to_index[cleaned_text] = len(unique_texts)
                unique_texts.append(cleaned_text)

            index_map.append(text_to_index[cleaned_text])

        return unique_texts, index_map

    @torch.inference_mode()
    def translate_batch(
        self,
        captions: list[str],
        src_lang: str = "eng_Latn",
        tgt_lang: str = "npi_Deva",
        batch_size: int = 32,
        max_input_length: int = 128,
        max_output_length: int = 128,
    ) -> list[str]:
        """
        Translates a list of string captions using length-sorted batching, greedy search, and deduplication.
        """
        if not captions:
            return []

        self.load_model()
        device = next(self.model.parameters()).device

        indexed_nonempty = [
            (index, text.strip())
            for index, text in enumerate(captions)
            if text and text.strip()
        ]

        if not indexed_nonempty:
            return [""] * len(captions)

        original_indices = [item[0] for item in indexed_nonempty]
        nonempty_texts = [item[1] for item in indexed_nonempty]

        unique_texts, index_mapping = self._deduplicate(nonempty_texts)

        sorted_unique_pairs = sorted(
            enumerate(unique_texts), key=lambda x: len(x[1])
        )
        sorted_indices = [p[0] for p in sorted_unique_pairs]
        sorted_texts = [p[1] for p in sorted_unique_pairs]

        self.tokenizer.src_lang = src_lang

        # Robust resolution for target language token ID across Fast/Slow NLLB tokenizers
        target_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        if (
            target_token_id is None
            or target_token_id == self.tokenizer.unk_token_id
        ):
            # Fallback check for dictionary attribute on legacy tokenizer instances
            lang_code_map = getattr(self.tokenizer, "lang_code_to_id", {})
            target_token_id = lang_code_map.get(tgt_lang, self.tokenizer.unk_token_id)

        if target_token_id == self.tokenizer.unk_token_id:
            raise ValueError(
                f"Unsupported NLLB target language code: {tgt_lang}"
            )

        translated_sorted = []

        for start in range(0, len(sorted_texts), batch_size):
            batch_texts = sorted_texts[start : start + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
            ).to(device)

            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=target_token_id,
                max_length=max_output_length,
                num_beams=1,
                do_sample=False,
            )

            translations = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
            )

            translated_sorted.extend(t.strip() for t in translations)

        translated_unique = [""] * len(unique_texts)
        for sorted_pos, orig_pos in enumerate(sorted_indices):
            translated_unique[orig_pos] = translated_sorted[sorted_pos]

        translated_nonempty = [
            translated_unique[idx] for idx in index_mapping
        ]

        translated = [""] * len(captions)
        for original_index, translation in zip(
            original_indices, translated_nonempty
        ):
            translated[original_index] = translation

        return translated

    def translate_json_payload(
        self,
        json_data: list[dict],
        src_lang: str = "eng_Latn",
        tgt_lang: str = "npi_Deva",
        batch_size: int = 32,
    ) -> list[dict]:
        """
        Extracts captions from structured subtitle JSON, translates them,
        and returns updated JSON maintaining exact timecodes.
        """
        raw_captions = [item.get("text", "") for item in json_data]

        translated_captions = self.translate_batch(
            raw_captions,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            batch_size=batch_size,
        )

        translated_json = []
        for item, translated_text in zip(json_data, translated_captions):
            updated_item = item.copy()
            updated_item["text"] = translated_text
            translated_json.append(updated_item)

        return translated_json

    def offload_from_vram(self) -> None:
        """Clears intermediate CUDA caches."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
