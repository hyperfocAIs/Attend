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

from importlib import import_module
import re
import yaml
import time
import threading
import traceback
import json
import nltk
from openai import OpenAI
from typing import List, Dict, Any
from services.manage_recording import AudioRecordingService
from services.event_system import EventEmitter, SpeechEvent
from functions.streamtts import stream_text
from .audio import AudioProcessor
from .tts import TTSProcessor

class InteractionManager:
    def __init__(self, config_path: str, recording_service: AudioRecordingService, audio_device_manager):
        """Initialize the interaction manager."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.recording_service = recording_service
        self.audio_device_manager = audio_device_manager
        self.events = EventEmitter()
        self.current_mode = None
        self.messages: List[Dict[str, str]] = []
        self.messages_tentative: List[Dict[str, str]] = []
        self.debug = False
        
        # State management
        self.speech_ended = threading.Event()
        self.json_processed = False
        self.current_response = ""
        
        # Pipeline control
        self.current_pipeline = None
        self.pipeline_lock = threading.Lock()
        
        # Audio processing
        self.audio_processor = AudioProcessor(audio_device_manager)
        self.audio_processor.queue_lock = threading.Lock()
        
        # TTS processing
        self.tts_processor = TTSProcessor(self.config)
        
        # Playback control
        self.playback_complete = threading.Event()
        self.playback_complete.set()
        
        # OpenAI LLM client
        text_config = self.config["server-text"]
        self.llm_client = OpenAI(
            api_key=text_config["key"],
            base_url=f"{text_config['host']}:{text_config['port']}/v1"
        )
        
        # Set up event listeners
        self.events.on(SpeechEvent.SPEECH_END_POTENTIAL, self._handle_speech_end_potential)
        self.events.on(SpeechEvent.FALSE_END, self._handle_false_end)
        self.events.on(SpeechEvent.SPEECH_ENDED, self._handle_speech_ended)
        self.events.on(SpeechEvent.NEW_TRANSCRIPTION, self._handle_transcription)
        
    def _log(self, message):
        """Print debug messages if debug mode is enabled."""
        if self.debug:
            print(f"[InteractionManager] {message}")
            
    def _handle_speech_end_potential(self, *args):
        """Handle potential end of speech."""
        self._log("Potential speech end detected")
        
    def _handle_false_end(self, *args):
        """Handle false end detection."""
        self._log("False end detected")
        # Abort current pipeline and clear queued audio
        with self.pipeline_lock:
            self.current_pipeline = None
            self.json_processed = False
            self.current_response = ""
        self.audio_processor.queued_audio = []
        
    def _handle_speech_ended(self, *args):
        """Handle confirmed end of speech."""
        self._log("Speech ended")

        # Play any queued audio responses
        self._play_queued_audio()


        self.speech_ended.set()
        self._log("_handle_speech_ended Playing queued audio and appending messages")
        
        # Process any pending responses
        if not self.json_processed:
            self._log("recording_service.manager.pipeline_state was confirmed and ~self.json_processed during _handle_speech_ended parsing accumelated response")
            self.messages = self.messages_tentative
            self.parse_accumulated_response()                

        
        
        # # Reset pipeline state
        # if self.recording_service and self.recording_service.manager:
        #     self._log("Resetting pipeline state after speech ended")
        #     self.recording_service.manager.pipeline_processing = False
        #     self.recording_service.manager.pipeline_state = None
        #     self.recording_service.manager.current_pipeline_id = None
            
        #     # Force a VAD reset to ensure new speech can be detected
        #     self.recording_service.manager.reset()
        
        # Clear completion flags
        self.json_processed = False
        self.current_response = ""
        
    def _play_queued_audio(self):
        """Play queued audio after LLM response processing."""
        self._log("Attempting to play queued audio")
        try:
            self.audio_processor.play_queued_audio(self.playback_complete)
            self._log("Queued audio playback initiated")
        except Exception as e:
            self._log(f"Error playing queued audio: {str(e)}")
            self._log(f"Traceback: {traceback.format_exc()}")

    
    def parse_accumulated_response(self):
        self.json_processed=True
        try:
            response_data = json.loads(self.current_response)
            outputs = response_data.get('outputs', {})

            if 'next_mode' in outputs:

                self._log(f"JSON parsed and found next_mode: {outputs['next_mode']}")

                next_mode = outputs['next_mode']
                activity_description = outputs['activity_description']
                module = import_module(f"modes.{next_mode}")      

                self._log(f"Imported module")

                 # Set the activity description in the module before switching modes
                if hasattr(module, 'activity_description'):
                    module.activity_description = activity_description

                self._log(f"Set module.activity_description = module.activity_description")
                
                self._log(f"Module schema: {module.schema}")

                self.set_mode(module)

                self._log(f"After set_mode, mode.schema: {mode.schema}")

                
                #self.activity_description = activity_description
            elif 'assistant_response' in outputs:
                self._log(f"JSON parsed and found assistant response: {outputs['assistant_response']}")
                
            else:
                raise ValueError("Invalid response format")

        except json.JSONDecodeError:
            self._log(f"Invalid JSON in current_response. current_response: {self.current_response}")
            
        except KeyError as e:
            self._log(f"Missing required key in response: {e}")        
        
        except Exception as e:
            self._log(f"Error parsing response: {str(e)}")
            

        
    def _handle_transcription(self, transcription):
        """Handle new transcription from speech."""
        if not transcription:
            return
            
        # Get the latest complete transcription
        text = transcription[0]["text"]
        self._log(f"Handling transcription: {text}")
        
        # Get current pipeline ID and check state
        pipeline_id = self.recording_service.manager.current_pipeline_id
        pipeline_state = self.recording_service.manager.pipeline_state
        
        if pipeline_state in ['cancelled', 'false']:
            self._log(f"Pipeline {pipeline_id} is no longer valid (state: {pipeline_state})")
            return
            
        # Setup tetative messages
        self.messages_tentative = self.messages.copy()
        self.messages_tentative.append({
            "role": "user",
            "content": text
        })
        
        # Stream LLM response and TTS
        try:
            self._log("Creating LLM completion")
            # self._log("Current messages:")
            # for msg in self.messages:
            #     self._log(f"Role: {msg['role']}, Content: {msg['content']}")
            # self._log("Current messages_tentative:")
            # for msg in self.messages_tentative:
            #     self._log(f"Role: {msg['role']}, Content: {msg['content']}")

            response = self.llm_client.chat.completions.create(
                model=self.config["server-text"]["model"],
                messages=self.messages_tentative,
                response_format={
                    "type": "json_schema",
                    "json_schema": self.response_schema
                },
                stream=True
            )
            
            # Initialize parameters for looping through the chunks
            accumulated_response = ""
            accumulated_assistant_response = ""
            response_type_known = False            
            assistant_response_start_reached = False
            assistant_response_end_reached = False
            # There will be at least one sentence if there is content.
            # Once we are on the second, we can send the first for TTS, and so on
            accumulated_assistant_response = ""            
            n_identified_sentences = 1 
            processed_sentences = set()
            
            for chunk in response: 
                # Check pipeline state before processing each chunk and return if needed
                current_state = self.recording_service.manager.pipeline_state
                if current_state in ['cancelled', 'false']:
                    self._log(f"Pipeline {pipeline_id} cancelled during processing")
                    return
                # For each incoming piece of response
                if chunk.choices[0].delta.content is not None:
                    # Accumulate it
                    content = chunk.choices[0].delta.content
                    #print(content)
                    accumulated_response += content
                    #self._log(f"Received chunk: {content}")
                    
                    # Check if we know the type of response we're getting yet
                    if not response_type_known:
                        # Use regex to match the pattern of the JSON response and account for whitespace
                        match = re.search(r'"outputs"\s*:\s*{\s*"(\w+)"', accumulated_response)
                        #print(match)
                        if match:
                            response_type = match.group(1)
                            response_type_known = True
                    # if we already know and it is the assistant_response we'll look for sentences.
                    elif response_type == "assistant_response":
                        # We need to check if we've gotten all the way to the text of the response
                        if not assistant_response_start_reached:
                            match = re.search(r'"outputs"\s*:\s*{\s*"assistant_response"\s*:\s*"', accumulated_response)
                            if match:
                                assistant_response_start_reached = True
                        # Once we have, parse, unless we've reached the end
                        elif not assistant_response_end_reached:                            
                            accumulated_assistant_response += content
                            #Check if we've reached the end
                            if re.search(r'.*"', accumulated_assistant_response):
                                # if so, mark that we're at the end
                                assistant_response_end_reached = True
                                
                                # and send the final sentence for tts
                                sentences = nltk.sent_tokenize(accumulated_assistant_response)
                                self._log(f"Sending final sentence '{sentences[n_identified_sentences - 1]}' for TTS streaming")
                                audio_data = self.tts_processor.process_tts(sentences[n_identified_sentences - 1])
                                self._log(f"Queing audio for {sentences[n_identified_sentences - 1]}")
                                self.audio_processor.queue_audio(audio_data)
                                # If the pipeline has been confirmed, play the queued audio
                                if self.recording_service.manager.pipeline_state == 'confirmed':
                                    self._log("Playing queued audio within _handle_transcription")
                                    self._play_queued_audio()
                            # if not at the end yet
                            else:
                                 sentences = nltk.sent_tokenize(accumulated_assistant_response)
                                 # If there are more sentences than previously, send the penultimate for TTS streaming
                                 if len(sentences) > n_identified_sentences:
                                    self._log(f"Accumulated assistant response so far: {accumulated_assistant_response}")
                                    audio_data = self.tts_processor.process_tts(sentences[n_identified_sentences - 1])
                                    self.audio_processor.queue_audio(audio_data)
                                    # If the pipeline has been confirmed, play the queued audio
                                    if self.recording_service.manager.pipeline_state == 'confirmed':
                                        self._play_queued_audio()
                                    n_identified_sentences += 1


            self._log(f"Finished processing response: {accumulated_response}")
            # Save the full response (not just the assistant response)
            self.current_response = accumulated_response
            # This response has not yet had its JSON parsed
            self.json_processed = False

            # Update tentative messages if the assistant said anything
            if accumulated_assistant_response != "":
                self.messages_tentative.append({
                    "role": "assistant",
                    "content": accumulated_assistant_response
                })

            self._log(f"Finished _handle_transcription loop. Current messages_tentative: {self.messages_tentative}")


            self._log(f"Checking if pipeline is confirmed within _handle_transcription")
            # if the pipeline has been confirmed, save them as actual messages,
            # play any remaining audio, and parse the response
            if self.recording_service.manager.pipeline_state == 'confirmed':              
                self._log("Pipeline confirmed within _handle_transcription")  
                self.messages = self.messages_tentative
                self._play_queued_audio()
                self.parse_accumulated_response()
                
                
            

            
            
        except Exception as e:
            self._log(f"Error in LLM/TTS processing: {str(e)}")
            self._log(f"Traceback: {traceback.format_exc()}")
        
    
            
    def set_mode(self, mode):
        """Set the current interaction mode."""
        self._log(f"Setting mode: {mode.__name__ if hasattr(mode, '__name__') else mode}")
        self.current_mode = mode
        
        # Initialize mode
        if hasattr(mode, 'before_first_turn'):
            self._log("Calling before_first_turn")
            # Pass manager instance to before_first_turn
            mode.before_first_turn.manager = self
            mode.before_first_turn()
            
        # Handle mode initialization
        if hasattr(mode, 'initialize'):
            self._log("Mode has initialization configuration")
            if isinstance(mode.initialize, dict):
                if "greeting" in mode.initialize:
                    self._log(f"Processing greeting: {mode.initialize['greeting']['text']}")
                    stream_text(
                        text=mode.initialize["greeting"]["text"], 
                        audio_manager=self.audio_device_manager,
                        speed=mode.initialize["greeting"]["speed"]
                    )
                    self.messages = [
                        {"role": "system", "content": mode.system_prompt},
                        {"role": "user", "content": "Let's get to it."},
                        {"role": "assistant", "content": mode.initialize["greeting"]["text"]}
                    ]
                    self._log("Greeting processed and messages initialized")
                elif "prompt" in mode.initialize:
                    self._log("Prompt initialization not yet implemented")
                    pass

        self.response_schema = mode.schema         

        if hasattr(mode, 'after_attend_turn'):
            self._log("Calling after_attend_turn")
            mode.after_attend_turn()
            
        # Store messages for potential rollback
        self.messages_tentative = self.messages.copy()
        self._log("Mode setup completed")
