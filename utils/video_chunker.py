import math
import subprocess
from pathlib import Path

class VideoChunker:
    """
    Splits video files into fixed-duration chunks using FFmpeg subprocess calls.
    """

    def __init__(self, output_dir: str = "temp_chunks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_duration(self, video_path: str) -> float:
        """Retrieves total duration of the video in seconds using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]

        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        return float(result.stdout.strip())

    def split_video(
        self, video_path: str, chunk_duration: int = 4
    ) -> list[dict]:
        """
        Splits video into chunks and returns list of metadata dicts containing:
        - chunk_path: File path to temporary video chunk
        - start_time: Start time in seconds
        - end_time: End time in seconds
        """
        total_duration = self.get_duration(video_path)
        num_chunks = math.ceil(total_duration / chunk_duration)
        chunk_metadata = []

        video_path_obj = Path(video_path)

        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = min((i + 1) * chunk_duration, total_duration)
            duration = end_time - start_time

            # Ignore residual tail chunks shorter than 0.2 seconds
            if duration < 0.2:
                continue

            chunk_filename = f"{video_path_obj.stem}_chunk_{i}.mp4"
            chunk_path = self.output_dir / chunk_filename

            cmd = [
                "ffmpeg",
                "-y",                             # Overwrite if exists
                "-ss", str(start_time),           # Fast input seeking
                "-i", str(video_path),
                "-t", str(duration),              # Precise segment length
                "-c:v", "libx264",                # Accurate re-encoding
                "-preset", "ultrafast",
                "-c:a", "aac",
                "-avoid_negative_ts", "make_zero",
                str(chunk_path),
            ]

            subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )

            # Validate that the sliced chunk was created and contains video data
            if chunk_path.exists() and chunk_path.stat().st_size > 0:
                chunk_metadata.append(
                    {
                        "chunk_path": str(chunk_path),
                        "start_time": start_time,
                        "end_time": end_time,
                    }
                )

        return chunk_metadata
