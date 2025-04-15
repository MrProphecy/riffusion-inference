# riffusion_cli.py
import argparse
import torchaudio
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.audio_file import wav_bytes_from_spectrogram_image
from riffusion.pipeline import build_pipeline

def main():
    parser = argparse.ArgumentParser(description="Generate audio from a prompt")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for audio generation")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save .wav output")

    args = parser.parse_args()

    pipeline = build_pipeline()
    converter = SpectrogramImageConverter()

    print(f"🎤 Prompt recibido: {args.prompt}")
    result = pipeline.run_image(prompt=args.prompt, denoising=0.75, guidance=7.0)

    image = result["spectrogram"]
    audio_bytes = wav_bytes_from_spectrogram_image(converter, image)

    with open(args.output_path, "wb") as f:
        f.write(audio_bytes)

    print(f"✅ Audio generado en: {args.output_path}")

if __name__ == "__main__":
    main()
