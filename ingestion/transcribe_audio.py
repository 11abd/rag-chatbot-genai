import whisper
from pathlib import Path


def transcribe_audio(audio_path: Path, model) -> str:
    """
    Transcribes audio file using Whisper.
    """
    result = model.transcribe(str(audio_path))
    return result["text"]


def transcribe_all(audio_dir: str, output_dir: str):
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Whisper base model...")
    model = whisper.load_model("base")

    for audio_file in audio_dir.glob("*.wav"):
        print(f"Transcribing: {audio_file.name}")
        text = transcribe_audio(audio_file, model)

        output_file = output_dir / f"{audio_file.stem}.txt"
        output_file.write_text(text, encoding="utf-8")

    print("\n✅ Transcription completed")


if __name__ == "__main__":
    transcribe_all(
        audio_dir="data/audio",
        output_dir="data/transcripts"
    )
