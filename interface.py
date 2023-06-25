
import base64
import logging
import sys

from brains import chatgpt
from session_manager import update_conversation, fix_typos_in_wake_word, is_user_talking_to_me
from speech_module.transcription import LiveTranscription

import os

import streamlit as st
from streamlit_chat import message

from llama_cpp import Llama

from speech_module.tts_model import TextToSpeechModel

def autoplay_audio(file_path="speech.wav", idx=0):
    print("Playing audio file: ", file_path)
    with open(file_path, "rb") as binary_audio:
        audio_bytes = binary_audio.read()

    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    audio_tag = f'<audio autoplay="true" src="data:audio/wav;base64,{audio_base64}">'
    st.markdown(audio_tag, unsafe_allow_html=True) 

def alpaca_model(conversation, model):
    response = model.create_chat_completion(conversation)
    return response['choices'][0]['message']['content']

def pipeline(tts, engine="gpt4"):
    print("Live speech recognition")

    with st.spinner("Loading"):

        if engine == "alpaca":
            logger.info('Loaded Alpaca-7B model...')
            llm_model = Llama(model_path="models/llm/ggml-model-q4_0.bin")
            logger.info('LLM loaded.')

        logger.info("Llama model loaded.")    

        live_transcription = LiveTranscription()
        live_transcription.start()
        logger.info("Started live transcription.")

        with open("system_prompt.txt", "r") as f:
            system_prompt = f.read()
        print(system_prompt)

        conversation = [{"role": "system", 
                        "content": system_prompt}]

    with open("wake_words.txt", "r") as f:
        wake_words = f.read().split("\n")

    st.title("Listening...")
    message("Hi I'm Alice, ask me anything! Just use a wake word in your sentence. I respond to Mark One, Alice, or Jarvis!", is_user=True, avatar_style="adventurer", seed="Whiskers")

    try:
        while True:
            transcript, sample_length, inference_time, confidence = live_transcription.get_last_text()
            print(f"{sample_length:.3f}s\t{inference_time:.3f}s\t{confidence}\t{transcript}")

            if is_user_talking_to_me(transcript, wake_words): 
                logger.info("User is talking to me!")

                transcript = fix_typos_in_wake_word(transcript, wake_words, "Alice")
                message(transcript, avatar_style="adventurer", seed="Trouble")

                with st.spinner("Generating audio: "):
                    update_conversation(conversation, "user", transcript)

                    if engine == "alpaca":
                        logger.info("Alpaca getting response")
                        response = alpaca_model(conversation, llm_model)
                        logger.info(response)

                    elif engine == "gpt4":
                        logger.info("GPT-4 getting response")
                        response = chatgpt(content=conversation) 
                        logger.info(response)

                    print("Prompt invoked: ", conversation) 
                    update_conversation(conversation, "assistant", response)
                    message(response, is_user=True, avatar_style="adventurer", seed="Whiskers")

                    speech_file = tts.tts_generator(response)
                    autoplay_audio(speech_file)

            if transcript == "quit": 
                message("Goodbye!", is_user=True, avatar_style="adventurer", seed="Whiskers")
                live_transcription.stop()
                exit()
            
            if transcript == "restart":
                live_transcription.stop()
                os.execvp(sys.executable, [os.environ.get("WORKDIR")+'/.venv/bin/python'] + sys.argv)

    except KeyboardInterrupt:
        live_transcription.stop()
        exit()

if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info('Starting Logger')

    logger.info("Loading TTS model...")
    tts = TextToSpeechModel()
    logger.info("TTS model loaded")

    pipeline(tts, engine="gpt4")
