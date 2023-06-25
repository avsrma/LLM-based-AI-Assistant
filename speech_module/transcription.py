import pyaudio
from torch import device
import webrtcvad
import numpy as np
import threading
import time
from sys import exit
from queue import  Queue

from speech_module.inference import Inference


class LiveTranscription():
    exit_event = threading.Event()

    def __init__(self, device_name="default"):
        self.device_name = device_name

    def stop(self):
        LiveTranscription.exit_event.set()
        self.recognition_input_queue.put("close")
        print("Stopping listening process")

    def start(self):
        self.recognition_output_queue = Queue()
        self.recognition_input_queue = Queue()

        self.recognition_process = threading.Thread(target=LiveTranscription.recognition_process, args=(self, self.recognition_input_queue, self.recognition_output_queue))
        
        self.recognition_process.start()
        
        time.sleep(5)  # start vad after recognition model is loaded
        self.vad_process = threading.Thread(target=LiveTranscription.vad_process, 
                                            args=(self.device_name, 
                                                  self.recognition_input_queue,
                                                  ))
        self.vad_process.start()

    def vad_process(device_name, recognition_input_queue):
        vad = webrtcvad.Vad()
        vad.set_mode(1)

        audio = pyaudio.PyAudio()
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        FRAME_DURATION = 30
        CHUNK = int(RATE * FRAME_DURATION / 1000)
        RECORD_SECONDS = 10

        microphones = LiveTranscription.list_microphones(audio)
        selected_input_device_id = LiveTranscription.get_input_device_id(device_name, microphones)

        stream = audio.open(input_device_index=selected_input_device_id,
                            format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

        frames = b''
        while True:
            if LiveTranscription.exit_event.is_set():
                break
            frame = stream.read(CHUNK, exception_on_overflow=False)
            is_speech = vad.is_speech(frame, RATE)
            if is_speech:
                frames += frame
            else:
                if len(frames) > 1:
                    recognition_input_queue.put(frames)
                frames = b''
        stream.stop_stream()
        stream.close()
        audio.terminate()

    def recognition_process(self, in_queue, output_queue):
        live_inference = Inference()

        print("\nListening to your voice\n")
        while True:
            audio_frames = in_queue.get()
            if audio_frames == "close":
                break

            float64_buffer = np.frombuffer(
                audio_frames, dtype=np.int16) / 32767
            start = time.perf_counter()

            transcript, confidence = live_inference.buffer_to_text(float64_buffer)
            transcript = transcript.lower()

            inference_time = time.perf_counter() - start
            sample_length = len(float64_buffer) / 16000

            if transcript != "":
                output_queue.put([transcript, sample_length, inference_time, confidence]) 

    @staticmethod
    def get_input_device_id(device_name, microphones):
        for device in microphones:
            if device_name in device[1]:
                return device[0]

    @staticmethod
    def list_microphones(pyaudio_instance):
        info = pyaudio_instance.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')

        result = []

        for i in range(0, numdevices):
            if (pyaudio_instance.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                name = pyaudio_instance.get_device_info_by_host_api_device_index(
                    0, i).get('name')
                result += [[i, name]]
        return result

    def get_last_text(self):
        return self.recognition_output_queue.get()
