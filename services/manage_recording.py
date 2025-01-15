# This source code is a part of Attend. Attend is a voice assistant that uses 
# very expensive algorithms to direct your attention... however you damn well please.
# Copyright (C) 2025 Scott Macdonell

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import yaml
#import math
import time
import pyaudio
import wave
import os
import numpy as np
import threading
from collections import deque
from typing import Callable
from openai import OpenAI
from silero_vad import load_silero_vad, VADIterator
from .audio_device_manager import AudioDeviceManager
from .event_system import EventEmitter, SpeechEvent

class RecordingManager:
    def __init__(self, config_path, audio_manager: AudioDeviceManager):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize event emitter
        self.events = EventEmitter()
        
        # Audio configuration
        self.chunk = self.config['client']['audio']['chunk']
        self.format = eval(self.config['client']['audio']['format'])
        self.channels = self.config['client']['audio']['channels']
        self.rate = self.config['client']['audio']['rate']
        self.buffer_seconds = self.config['client']['audio']['buffer_seconds']
        self.speech_start_threshold = self.config['client']['audio']['speech_start_threshold']
        self.speech_end_threshold = self.config['client']['audio']['speech_end_threshold']

        # Initialize audio recording components
        self.buffer_size = int(self.rate / self.chunk * self.buffer_seconds)
        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.chunk_times = deque(maxlen=self.buffer_size)  # Track timestamp of each chunk
        self.buffer_lock = threading.Lock()  # Lock for thread-safe buffer access
        
        # Use provided audio manager and get input stream
        self.audio_manager = audio_manager
        self.stream = self.audio_manager.input_stream
        if not self.stream:
            raise ValueError("AudioDeviceManager must have an initialized input stream")

        # Initialize VAD components
        self.model = load_silero_vad()
        self.vad_iterator = VADIterator(self.model, sampling_rate=self.rate)

        # Speech state tracking
        self.speech_start_time = None
        self.speech_end_time = None
        self.speech_start_potential = False
        self.speech_end_potential = False
        self.speech_started = False
        self.speech_detected = False
        self.speech_ended = False

        # Latest transcription storage
        self.latest_transcription = None

        # Thread control
        self.is_recording = False
        self.recording_thread = None

        # Pipeline state tracking
        self.current_pipeline_id = None
        self.pipeline_state = None  # None, 'false', 'confirmed', 'cancelled'
        self.pipeline_processing = False

    def _continuous_recording(self):
        """Background thread function for continuous audio recording and VAD processing."""
        while self.is_recording:
            #print("Recording thread still running...") 
            try:
                # Verify stream is still active
                if not self.stream or not self.stream.is_active():
                    raise RuntimeError("Audio input stream is not active")
                    
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                current_time = time.time()
                
                # Thread-safe buffer update
                with self.buffer_lock:
                    self.audio_buffer.append(data)
                    self.chunk_times.append(current_time)

                # Process audio chunk with VAD in the same thread
                audio_np = np.frombuffer(data, dtype=np.int16)
                audio_float32 = audio_np.astype(np.float32) / 32768.0

                # Process with VAD
                vad_result = self.vad_iterator(audio_float32, return_seconds=True)
             #   print(f"VAD - Result: {vad_result}, Started: {self.speech_started}, Start Potential: {self.speech_start_potential}, End Potential: {self.speech_end_potential}")

                ## VAD state machine logic
                # Handle potential start
                if isinstance(vad_result, dict) and 'start' in vad_result and not self.speech_started:
                    self.speech_start_time = current_time
                    self.speech_start_potential = True
                    self.speech_end_potential = False
                    self.events.emit(SpeechEvent.SPEECH_START_POTENTIAL)
                    print(f"Pipeline state check - ID: {self.current_pipeline_id}, State: {self.pipeline_state}, Processing: {self.pipeline_processing}")
                    # Only reset pipeline tracking if there isn't a confirmed pipeline still processing
                    if not (self.current_pipeline_id and self.pipeline_state == 'confirmed' and self.pipeline_processing):
                        self.current_pipeline_id = None
                        self.pipeline_state = None
                        self.pipeline_processing = False

                #Speech Started
                elif vad_result is None and self.speech_start_potential and not self.speech_started:
                    if current_time - self.speech_start_time >= self.speech_start_threshold:
                        self.speech_started = True
                        self.speech_start_potential = False
                        self.speech_detected = True
                        self.events.emit(SpeechEvent.SPEECH_STARTED)
                
                # Handle potential end
                elif isinstance(vad_result, dict) and 'end' in vad_result and self.speech_started:
                    if not self.speech_end_potential:
                        self.speech_end_time = current_time
                        self.speech_end_potential = True
                        self.events.emit(SpeechEvent.SPEECH_END_POTENTIAL)
                        
                        ## Pipeline management
                        # If there's a non-confirmed pipeline, cancel it
                        # if self.current_pipeline_id and self.pipeline_state not in ['confirmed', None]:
                        #     self.pipeline_state = 'cancelled'
                        # TODO need to improve pipeline tracking just in case new speech ends before prior LLM responses finishes.
                        self.current_pipeline_id = time.time()
                        self.pipeline_state = None
                        self.pipeline_processing = False

                # Handle speech ended
                elif vad_result is None and self.speech_started and self.speech_end_potential:
                    if current_time - self.speech_end_time >= self.speech_end_threshold:
                        self.speech_started = False
                        self.speech_end_potential = False
                        if self.current_pipeline_id:
                            self.pipeline_state = 'confirmed'
                        self.speech_ended = True
                        self.events.emit(SpeechEvent.SPEECH_ENDED)

                # Handle False Start
                elif isinstance(vad_result, dict) and 'end' in vad_result and self.speech_start_potential and not self.speech_started:
                    # False Start
                    self.speech_start_potential = False
                    self.speech_start_time = None
                    self.events.emit(SpeechEvent.FALSE_START)

                # Handle False End
                elif isinstance(vad_result, dict) and 'start' in vad_result and self.speech_started and self.speech_end_potential:
                    # False End
                    self.speech_end_potential = False
                    self.speech_end_time = None
                    # If there's a non-confirmed pipeline mark it as false, cancel it
                    if self.current_pipeline_id and not self.pipeline_state == 'confirmed':
                        self.pipeline_state = 'false'
                    self.events.emit(SpeechEvent.FALSE_END)


            except Exception as e:
                print(f"Error in recording thread: {str(e)}")
                self.is_recording = False
                break

    def start_recording(self):
        """Start the background recording thread."""
        if not self.stream or not self.stream.is_active():
            raise RuntimeError("Audio input stream is not active")
            
        if not self.is_recording:
            self.is_recording = True
            self.recording_thread = threading.Thread(target=self._continuous_recording)
            self.recording_thread.daemon = True
            self.recording_thread.start()

    def stop_recording(self):
        """Stop the background recording thread."""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()
            self.recording_thread = None

    def get_latest_chunk(self):
        """Get the latest audio chunk and its timestamp from the buffer."""
        with self.buffer_lock:
            if len(self.audio_buffer) > 0:
                return self.audio_buffer[-1], self.chunk_times[-1]
            return None, None

    # Logic Flow Table:
    # | Flow                | vad_result                     | self.speech_start_potential | self.speech_started | self.speech_end_potential | Actions                                                                                                                                                                                         |
    # | ------------------- | ------------------------------ | --------------------------- | ------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    # | Standard            | {'start': time}                | FALSE                       | FALSE               | FALSE                     | Set self.speech_start_time = current_time, self.speech_start_potential = True                                                                                                                   |
    # | Standard            | None                           | TRUE                        | FALSE               | FALSE                     | If current_time - speech_start_time >= speech_start_threshold:<br>we're now in a speech segment. So, self.speech_started = True, self.speech_start_potential=False, self.speech_detected = True |
    # | Standard            | None                           | FALSE                       | TRUE                | FALSE                     | Nothing, await next significant event                                                                                                                                                           |
    # | Standard            | {'end': time}                  | FALSE                       | TRUE                | FALSE                     | Set self.speech_end_time = current_time, self.speech_end_potential = True                                                                                                                       |
    # | Standard            | None                           | FALSE                       | TRUE                | TRUE                      | If current_time - speech_end_time >= speech_end_threshold:<br>Our speech segment ended. So, self.speech_started = Fase, self.speech_end_potential=False, self.speech_ended = True               |
    # | False Start         | {'end': time}                  | TRUE                        | FALSE               | FALSE                     | Speech did not last long enough to actually start a speech segment.<br>So, set self.speech_start_potential = False, self.speech_start_time = None                                               |
    # | False End           | {'start': time}                | FALSE                       | TRUE                | TRUE                      | Speech did not stop long enough to actually end a speech segment.<br>So, set self.speech_end_potential = False, self.speech_end_time = None                                                     |
    # | Blip before startup | {'start': time1, 'end': time2} | FALSE                       | FALSE               | FALSE                     | Nothing, await next significant event                                                                                                                                                           |
    # | Blip before end     | {'start': time1, 'end': time2} | FALSE                       | TRUE                | TRUE                      | Speech may be restarting. Give additional time to determine:<br>So, set self.speech_end_time = current_time                                                                                     |

    def save_speech(self, filename="to-process-for-STT.wav"):
        if not os.path.exists('temp'):
            os.makedirs('temp')

        file_path = os.path.join("temp", filename)

        # Mark pipeline as processing at start of save
        if self.current_pipeline_id:
            self.pipeline_processing = True

        # Find the buffer indices that correspond to our speech segment
        start_time = self.speech_start_time
        end_time = self.speech_end_time if self.speech_end_time else time.time()

        # Thread-safe access to buffers
        with self.buffer_lock:
            # Convert chunk_times to list for easier indexing
            chunk_times_list = list(self.chunk_times)
            audio_buffer_list = list(self.audio_buffer)

            # Find the indices that correspond to our speech segment
            start_idx = 0
            end_idx = len(chunk_times_list)

            for i, chunk_time in enumerate(chunk_times_list):
                if chunk_time >= start_time:
                    start_idx = max(0, i - int((self.speech_end_threshold*self.rate)//self.chunk))  # Include all chunks within an end_threshold before
                    break

            for i, chunk_time in enumerate(chunk_times_list[start_idx:], start_idx):
                if chunk_time > end_time:
                    end_idx = min(len(chunk_times_list), i + 1)  # Include one chunk after for smooth transition
                    break

            # Extract only the audio data for our speech segment
            speech_data = audio_buffer_list[start_idx:end_idx]

        wf = wave.open(file_path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio_manager.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(speech_data))
        wf.close()

        return file_path

    def process_stt(self, audio_path):
        host = self.config['server-stt']['host']
        port = self.config['server-stt']['port']
        key = self.config['server-stt']['key']
        model = self.config['server-stt']['model']
        base_url = f"{host}:{port}/v1/"
        
        try:
            client = OpenAI(api_key=key, base_url=base_url)
            with open(audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model=model,
                    file=audio_file
                )
                self.latest_transcription = [{"start": 0, "end": 0, "text": transcript.text}]
                self.events.emit(SpeechEvent.NEW_TRANSCRIPTION, self.latest_transcription)
                return self.latest_transcription
        except Exception as e:
            print(f"Error communicating with STT server: {str(e)}")
            return None

    def get_speech_duration(self):
        if self.speech_start_time and self.speech_end_time:
            return self.speech_end_time - self.speech_start_time
        return None

    def reset(self):
        """Reset the VAD state machine while preserving the audio buffer."""
        self.vad_iterator.reset_states()
        self.speech_start_time = None
        self.speech_end_time = None
        self.speech_start_potential = False
        self.speech_end_potential = False
        self.speech_started = False
        self.speech_detected = False
        self.speech_ended = False

    def close(self):
        """Close the recording manager."""
        self.stop_recording()  # Stop the recording thread
        self.reset()

    def get_latest_transcription(self):
        return self.latest_transcription

    def add_event_listener(self, event: SpeechEvent, callback: Callable) -> None:
        """Add an event listener for a specific speech event."""
        self.events.on(event, callback)

    def remove_event_listener(self, event: SpeechEvent, callback: Callable) -> None:
        """Remove an event listener for a specific speech event."""
        self.events.off(event, callback)


class AudioRecordingService:
    def __init__(self, config_path="attend_config.yaml", debug=False, audio_manager: AudioDeviceManager = None):
        """
        Initialize the audio recording service.
        
        Args:
            config_path (str): Path to the configuration file
            debug (bool): Whether to print debug output
            audio_manager (AudioDeviceManager): Instance of AudioDeviceManager to use
        """
        if audio_manager is None:
            raise ValueError("AudioDeviceManager instance must be provided")
            
        if not audio_manager.input_stream:
            raise ValueError("AudioDeviceManager must have an initialized input stream")
            
        self.manager = RecordingManager(config_path, audio_manager)
        self.debug = debug
        self.is_running = False
        self.previous_speech_detected = False
        self.previous_speech_end_potential = False
        self.processing_thread = None

    def _log(self, message):
        """Print debug messages if debug mode is enabled."""
        if self.debug:
            print(message)

    def _process_loop(self):
        """Main processing loop for audio recording and transcription."""
        try:
            while self.is_running:
                # Monitor speech detection states
                if self.manager.speech_detected and not self.previous_speech_detected:
                    self._log("Speech detected!")
                self.previous_speech_detected = self.manager.speech_detected

                # Handle both potential ends and false ends
                if (self.manager.speech_end_potential and not self.previous_speech_end_potential):
                    self._log("Speech segment potentially ended. Starting processing...")
                    pipeline_id = self.manager.current_pipeline_id
                    audio_path = self.manager.save_speech()

                    # Only process if pipeline hasn't been cancelled/marked false
                    if self.manager.pipeline_state not in ['cancelled', 'false']:
                        transcription = self.manager.process_stt(audio_path)
                        self._log(f"Transcription complete for pipeline {pipeline_id}")
                    
                self.previous_speech_end_potential = self.manager.speech_end_potential

                if self.manager.speech_ended:
                    self._log("Speech ended. Resetting state machine...")
                    self.manager.reset()

                time.sleep(0.01)  # Small delay to prevent busy-waiting

        except Exception as e:
            self._log(f"Error in processing loop: {str(e)}")
            self.stop()

    def start(self):
        """Start the audio recording service."""
        if not self.is_running:
            self.is_running = True
            self.manager.start_recording()
            self._log("Audio recording service started")
            self.processing_thread = threading.Thread(target=self._process_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()

    def stop(self):
        """Stop the audio recording service."""
        if self.is_running:
            self.is_running = False
            if self.processing_thread:
                self.processing_thread.join()
            self.manager.close()
            self._log("Audio recording service stopped")

    def get_latest_transcription(self):
        """Get the latest transcription from the manager."""
        return self.manager.get_latest_transcription()

    def add_event_listener(self, event: SpeechEvent, callback: Callable) -> None:
        """Add an event listener for a specific speech event."""
        self.manager.add_event_listener(event, callback)

    def remove_event_listener(self, event: SpeechEvent, callback: Callable) -> None:
        """Remove an event listener for a specific speech event."""
        self.manager.remove_event_listener(event, callback)
