import soundfile as sf
import torch
from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2Processor, WhisperProcessor, WhisperForConditionalGeneration

from speech_module.stt_model import SpeechToTextModel


class Inference():
    def __init__(self):
        self.stt = SpeechToTextModel()

    def buffer_to_text(self, audio_buffer):
        if(len(audio_buffer)==0):
            return ""

        inputs = self.stt.processor(torch.tensor(audio_buffer), 
                                sampling_rate=16000, 
                                return_tensors="pt", 
                                padding=True)

        with torch.no_grad():
            logits = self.stt.model(inputs.input_values, 
                                attention_mask=inputs.attention_mask).logits


        if hasattr(self.stt.processor, 'decoder') and self.stt.use_autoprocessor:
            transcript = \
                self.stt.processor.decode(logits[0].cpu().numpy(),
                                        hotwords=self.stt.hotwords,
                                        output_word_offsets=True, 
                                    )
            confidence = transcript.lm_score / len(transcript.text.split(" "))
            transcript = transcript.text 

        else:
            predicted_ids = torch.argmax(logits, dim=-1)
            transcript = self.stt.processor.batch_decode(predicted_ids)[0]
            confidence = self.confidence_score(logits,predicted_ids)

        return transcript, confidence 

    def confidence_score(self, logits, predicted_ids):

        scores = torch.nn.functional.softmax(logits, dim=-1)
        pred_scores = scores.gather(-1, predicted_ids.unsqueeze(-1))[:, :, 0]
        mask = torch.logical_and(
            predicted_ids.not_equal(self.stt.processor.tokenizer.word_delimiter_token_id), 
            predicted_ids.not_equal(self.stt.processor.tokenizer.pad_token_id))

        character_scores = pred_scores.masked_select(mask)
        total_average = torch.sum(character_scores) / len(character_scores)

        return total_average