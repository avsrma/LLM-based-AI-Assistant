import time
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile

class TextToSpeechModel():
    def __init__(self) -> None:
        self.embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation", cache_dir="models/text_to_speech")
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts", cache_dir="models/text_to_speech")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts", cache_dir="models/text_to_speech")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", cache_dir="models/text_to_speech")

    def tts_generator(self, input_text, use_gpu=False, device="cpu") -> str:
        print("input_text ", input_text)
        st = time.time()

        inputs = self.processor(
            text=input_text, 
            return_tensors="pt") 

        if use_gpu and torch.backends.mps.is_available():
            print("Using GPU") 
            device = torch.device("mps")
            self.model = self.model.to(device)
            self.vocoder = self.vocoder.to(device)
            inputs = inputs.to(device)

        # # load xvector containing speaker's voice characteristics from a dataset
        # embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embeddings = torch.tensor(self.embeddings_dataset[7306]["xvector"], device=device).unsqueeze(0)

        speech = self.model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=self.vocoder)
        out_file_path = f"saved_audio/speech{time.time()}.wav"

        soundfile.write(out_file_path, 
                speech.cpu().numpy(), 
                samplerate=16000)
        et = time.time()
        print("Time:", et-st)
        
        return out_file_path

# tts = TextToSpeechModel()
# tts.tts_generator("This is some random text.")

# def voice_generator(input_text):
#     tts = TextToSpeechModel()
#     return tts.tts_generator(input_text)

# print(voice_generator("some text randoom"))