import soundfile as sf
import torch
import resampy
import time
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline

# Load model and processor
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# print(device)
#processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
#model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(device)

# model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")

# pipe = pipeline(
#   "automatic-speech-recognition",
#   model="openai/whisper-large-v2",
#   chunk_length_s=30,
#   device=device,
# )



# pipe = pipeline(
#   "automatic-speech-recognition",
#   model="facebook/wav2vec2-base-960h",
#   device=device,
# )

def conv_sr(audio_file_path, desired_sr=16000):
    audio, sr = sf.read(audio_file_path) #sampling rate aligned with whisper
    if sr != desired_sr:
        return resampy.resample(audio.T, sr, desired_sr).T #transposing audio to (len, channels) to fit resampy
    return audio.T


def test_models(audio_file_path, mod_list, device):
    #audio = conv_sr(audio_file_path)[:, 0]
    audio = conv_sr(audio_file_path).mean(axis=1)  # One channel only for HF pipeline.
    for mod in mod_list:
        s = time.time()
        pipe = pipeline(
          "automatic-speech-recognition",
          model=mod,
          device=device
        )
        pred = pipe(audio, batch_size=1, return_timestamps="word", chunk_length_s=30, stride_length_s=(4,4))["text"]
        #TODO: test stride lengths
        dur = time.time() - s
        with open(f"{mod.split('/')[-1]}_{audio_file_path.split('.')[0]}.txt", "a+") as f:
            print("Duration:", str(dur), file=f)
            print(pred,file=f)

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aud_file="hmm.mp3"
    # aud_file = "10316_2.wav"

    # models = ["facebook/wav2vec2-base-960h", "openai/whisper-large-v2", "openai/whisper-medium"]
    models = ["facebook/wav2vec2-base-960h", "openai/whisper-medium", "openai/whisper-small", "openai/whisper-tiny", "openai/whisper-large"]

    test_models(aud_file, models, device)
# input_features = processor(audio, sampling_rate=desired_sr, return_tensors="pt").input_features.to(device)

# # Generate token ids
# predicted_ids = model.generate(input_features)

# # Decode token ids to text
# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
# print(transcription)
