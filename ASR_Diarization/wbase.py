import torch
import time
import soundfile as sf
from transformers import pipeline
from audio_utils import *
from diarize import *

class ASRPipe:
    def __init__(self, path, mod_name, device):
        self.path = path
        self.mod_name = mod_name
        self.device = device


    def run_pipe(self, split=True):
        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.mod_name,
            device=self.device
        )
        if split:
            mic1, mic2 = split_wav(self.path)  # One channel audio only for HF pipeline.
            pred_mic1 = pipe(mic1, batch_size=20, return_timestamps=True, chunk_length_s=30, stride_length_s=(4, 4))
            pred_mic2 = pipe(mic2, batch_size=20, return_timestamps=True, chunk_length_s=30, stride_length_s=(4, 4))
            return list(pred_mic1["chunks"]) + list(pred_mic2["chunks"])
        else:
            aud, sr = sf.read(self.path)
            audio =  aud.mean(axis=1)
            pred = pipe(audio, batch_size=20, return_timestamps=True, chunk_length_s=30, stride_length_s=(4, 4))
            return list(pred["chunks"])


    def diarize(self):
        pred = self.run_pipe()
        get_diarization(self.path)

        # Produces a list of tags for each word
    def store_output(self, pred, dur=None):
        seg_list = sorted(pred, key = lambda x: x["timestamp"][0])
        with open(f"{self.mod_name.split('/')[-1]}_{self.path.split('.')[0]}.txt", "w") as f:
            if dur:  # This shows how long the model took to process the file, only for testing.
                print(dur, file=f)
            for chunk in seg_list:
                print(chunk, file=f)

def test_models(audio_path, mod_list, device, record_model_time=True):
    #audio = conv_sr(audio_file_path)[:, 0]
    for mod in mod_list:
        s = time.time()
        piper = ASRPipe(audio_path, mod, device)
        try:
            pred = piper.run_pipe()
        except torch.cuda.OutOfMemoryError:
            pred = "Not enough GPU memory"
        #TODO: test stride lengths
        if record_model_time:
            dur = time.time() - s
        else:
            dur = None
        piper.store_output(pred, dur)

if __name__=="__main__":
    assert torch.cuda.is_available(), "CUDA Required for this program"
    dev = torch.device("cuda")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # aud_file="hmm.mp3"
    # aud_file = "10316_2.wav"
    aud_file = "14063_2.wav"

    # models = ["facebook/wav2vec2-base-960h", "openai/whisper-large-v2", "openai/whisper-medium"]
    # models = ["facebook/wav2vec2-base-960h", "openai/whisper-medium", "openai/whisper-small"]
    models = ["openai/whisper-tiny"]
    #asr_pipe = ASRPipe(aud_file, mod_name)
    test_models(aud_file, models, dev)
# input_features = processor(audio, sampling_rate=desired_sr, return_tensors="pt").input_features.to(device)

# # Generate token ids
# predicted_ids = model.generate(input_features)

# # Decode token ids to text
# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
# print(transcription)
