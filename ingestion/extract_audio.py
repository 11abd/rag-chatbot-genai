import ffmpeg
from pathlib import Path


def extract_audio_from_mp4(mp4_path: Path, output_dir: Path):
    """
    Extracts audio from an MP4 file and saves as WAV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / f"{mp4_path.stem}.wav"

    (
        ffmpeg
        .input(str(mp4_path))
        .output(str(audio_path), ac=1, ar=16000)
        .overwrite_output()
        .run(quiet=True)
    )

    return audio_path


def process_all_videos(video_dir: str, output_dir: str):
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)

    for video_file in video_dir.glob("*.mp4"):
        print(f"Extracting audio from: {video_file.name}")
        extract_audio_from_mp4(video_file, output_dir)

    print("\n✅ Audio extraction completed")


if __name__ == "__main__":
    process_all_videos(
        video_dir="data/audio",
        output_dir="data/audio"
    )
