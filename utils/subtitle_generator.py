from pathlib import Path

class SubtitleGenerator:
    """
    Formats timestamped caption data into WebVTT and SRT subtitle files.
    """

    @staticmethod
    def _format_timestamp(seconds: float, vtt_format: bool = True) -> str:
        """
        Converts seconds (float) into WebVTT (HH:MM:SS.mmm) or SRT (HH:MM:SS,mmm) timecode.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds - int(seconds)) * 1000))

        # Adjust for overflow from rounding
        if millis >= 1000:
            millis = 0
            secs += 1
            if secs >= 60:
                secs = 0
                minutes += 1
                if minutes >= 60:
                    minutes = 0
                    hours += 1

        separator = "." if vtt_format else ","
        return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"

    def build_payload(
        self, chunks_metadata: list[dict], captions: list[str]
    ) -> list[dict]:
        """
        Combines chunk timing metadata with generated/translated captions.

        Input chunks_metadata format: [{'start_time': 0.0, 'end_time': 5.0}, ...]
        Returns structured list of caption objects.
        """
        payload = []
        for meta, text in zip(chunks_metadata, captions):
            clean_text = text.strip()
            if clean_text:  # Skip empty captions
                payload.append(
                    {
                        "start_time": meta["start_time"],
                        "end_time": meta["end_time"],
                        "text": clean_text,
                    }
                )
        return payload

    def export_vtt(self, payload: list[dict], output_path: str) -> str:
        """
        Generates a WebVTT (.vtt) file from a structured payload list.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        vtt_lines = ["WEBVTT\n"]

        for idx, item in enumerate(payload, start=1):
            start_tc = self._format_timestamp(
                item["start_time"], vtt_format=True
            )
            end_tc = self._format_timestamp(item["end_time"], vtt_format=True)

            vtt_lines.append(f"{idx}")
            vtt_lines.append(f"{start_tc} --> {end_tc}")
            vtt_lines.append(f"{item['text']}\n")

        content = "\n".join(vtt_lines)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return output_path

    def export_srt(self, payload: list[dict], output_path: str) -> str:
        """
        Generates an SRT (.srt) file from a structured payload list.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        srt_lines = []

        for idx, item in enumerate(payload, start=1):
            start_tc = self._format_timestamp(
                item["start_time"], vtt_format=False
            )
            end_tc = self._format_timestamp(
                item["end_time"], vtt_format=False
            )

            srt_lines.append(f"{idx}")
            srt_lines.append(f"{start_tc} --> {end_tc}")
            srt_lines.append(f"{item['text']}\n")

        content = "\n".join(srt_lines)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return output_path
