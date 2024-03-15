import soundfile as sf
import torch
import resampy
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline

# Load model and processor
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(device)

# model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")

pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-large-v2",
  chunk_length_s=30,
  device=device,
)

# Load local .wav file
# audio_file_path = "10316_2.wav"
audio_file_path="hmm.mp3"
audio, sr = sf.read(audio_file_path) #sampling rate aligned with whisper
desired_sr = 16000
if sr != desired_sr:
    resampled_audio = resampy.resample(audio.T, sr, desired_sr)
else:
    resampled_audio = audio
audio = resampled_audio.T[:,0]
# Preprocess the audio file
print(audio.shape)
pred = pipe(audio, batch_size=10000, return_timestamps=True)["chunks"]
print(pred)
# input_features = processor(audio, sampling_rate=desired_sr, return_tensors="pt").input_features.to(device)

# # Generate token ids
# predicted_ids = model.generate(input_features)

# # Decode token ids to text
# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
# print(transcription)
