import  torch
assert torch.cuda.is_available(), "GPU required to run this program"

from whisperplus import (
    ASRDiarizationPipeline,
    download_and_convert_to_mp3,
    format_speech_to_dialogue,
)
# vid_path = "https://youtu.be/QmPLGt5rd_k"
#vid_path = "https://youtu.be/X15o2sG8HF4"
#audio_path = download_and_convert_to_mp3(vid_path)
audio_path = "10316_2.wav"

device = "cuda"


pipeline = ASRDiarizationPipeline.from_pretrained(
    #asr_model="openai/whisper-medium",
    asr_model="openai/whisper-tiny",
    diarizer_model="pyannote/speaker-diarization",
    use_auth_token="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",
    chunk_length_s=30,
    device=device,
)

output_text = pipeline(audio_path, num_speakers=2, min_speaker=1, max_speaker=2)
dialogue = format_speech_to_dialogue(output_text)
with open("output_dialog.txt", "w") as f:
    print(dialogue, file=f)