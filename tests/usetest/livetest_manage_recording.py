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



import os
import sys
import time
from services.audio_device_manager import AudioDeviceManager
from services.manage_recording import AudioRecordingService
from services.event_system import SpeechEvent

def test_recording():
    # Initialize audio device manager
    config_path = "attend_config.yaml"
    print("\nInitializing audio device manager...")
    audio_manager = AudioDeviceManager(config_path, debug=True)
    
    try:
        # Initialize streams
        print("Initializing audio streams...")
        audio_manager.initialize_streams()
        
        # Create recording service
        print("Creating recording service...")
        recording_service = AudioRecordingService(
            config_path=config_path,
            debug=True,
            audio_manager=audio_manager
        )
        
        # Set up event listeners
        def on_speech_start_potential(*args):
            print("\nPotential speech detected...")
            
        def on_speech_started(*args):
            print("Speech started!")
            
        def on_speech_end_potential(*args):
            print("Speech potentially ending...")
            
        def on_speech_ended(*args):
            print("Speech ended!")
            
        def on_false_start(*args):
            print("False start detected - speech was too short")
            
        def on_false_end(*args):
            print("False end detected - speech continuing")
            
        def on_new_transcription(transcription, *args):
            print(f"New transcription received: {transcription[0]['text']}")
        
        # Register event listeners
        recording_service.add_event_listener(SpeechEvent.SPEECH_START_POTENTIAL, on_speech_start_potential)
        recording_service.add_event_listener(SpeechEvent.SPEECH_STARTED, on_speech_started)
        recording_service.add_event_listener(SpeechEvent.SPEECH_END_POTENTIAL, on_speech_end_potential)
        recording_service.add_event_listener(SpeechEvent.SPEECH_ENDED, on_speech_ended)
        recording_service.add_event_listener(SpeechEvent.FALSE_START, on_false_start)
        recording_service.add_event_listener(SpeechEvent.FALSE_END, on_false_end)
        recording_service.add_event_listener(SpeechEvent.NEW_TRANSCRIPTION, on_new_transcription)
        
        # Start recording service
        print("\nStarting recording service... (Press Ctrl+C to stop)")
        recording_service.start()
        
        # Keep the script running
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping recording...")
            
    finally:
        # Cleanup
        print("Cleaning up...")
        if 'recording_service' in locals():
            recording_service.stop()
        audio_manager.terminate()
        print("Done!")

if __name__ == "__main__":
    test_recording()
