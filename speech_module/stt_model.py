import torch
from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM

class SpeechToTextModel():
    def __init__(self, 
                 model_name="maxidl/wav2vec2-large-xlsr-german", #"facebook/wav2vec2-large-960h-lv60-self", # "facebook/wav2vec2-large-robust-ft-swbd-300h", 
                 hotwords=["Hello"],
                 use_gpu=False, 
                 use_autoprocessor=True, 
                 cache_dir="models/speech_to_text"):

        self.model_name = model_name
        self.hotwords = hotwords
        self.use_gpu = use_gpu
        self.use_autoprocessor = use_autoprocessor
        self.cache_dir = cache_dir

        if use_autoprocessor: 
            self.processor = AutoProcessor.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        else:
            self.processor = Wav2Vec2ProcessorWithLM.from_pretrained(self.model_name, cache_dir=self.cache_dir)

        self.model = AutoModelForCTC.from_pretrained(self.model_name, ctc_loss_reduction="mean", cache_dir=self.cache_dir)
    
        if self.use_gpu and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            # self.model.to(self.device)

# stt = SpeechToTextModel()
# print(stt.processor)
