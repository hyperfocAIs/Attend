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


import threading
import time
from typing import Optional
from services.manage_recording import AudioRecordingService
from services.event_system import SpeechEvent
from .manager import InteractionManager

class InteractionService:
    def __init__(self, config_path="attend_config.yaml", debug=False, recording_service: Optional[AudioRecordingService] = None, audio_device_manager = None):
        """
        Initialize the interaction service.
        
        Args:
            config_path (str): Path to the configuration file
            debug (bool): Whether to print debug output
            recording_service (AudioRecordingService): Instance of AudioRecordingService to use
            audio_device_manager: Instance of AudioDeviceManager to use
        """
        if recording_service is None:
            raise ValueError("AudioRecordingService instance must be provided")
            
        if audio_device_manager is None:
            raise ValueError("AudioDeviceManager instance must be provided")
            
        self.debug = debug
        self.manager = InteractionManager(config_path, recording_service, audio_device_manager)
        self.manager.debug = debug  # Pass debug setting to manager
        self.is_running = False
        self.processing_thread = None
        
        # Set up event listeners
        self.recording_service = recording_service
        self.recording_service.add_event_listener(
            SpeechEvent.NEW_TRANSCRIPTION, 
            self.manager._handle_transcription
        )
        self.recording_service.add_event_listener(
            SpeechEvent.SPEECH_END_POTENTIAL,
            self.manager._handle_speech_end_potential
        )
        self.recording_service.add_event_listener(
            SpeechEvent.FALSE_END,
            self.manager._handle_false_end
        )
        self.recording_service.add_event_listener(
            SpeechEvent.SPEECH_ENDED,
            self.manager._handle_speech_ended
        )
        
        self._log("InteractionService initialized")
        
    def _log(self, message):
        """Print debug messages if debug mode is enabled."""
        if self.debug:
            print(f"[InteractionService] {message}")
            
    def _process_loop(self):
        """Main processing loop for interaction service."""
        try:
            while self.is_running:
                # Process any pending interactions
                time.sleep(0.01)  # Small delay to prevent busy-waiting
                
        except Exception as e:
            self._log(f"Error in processing loop: {str(e)}")
            self.stop()
            
    def start(self):
        """Start the interaction service."""
        if not self.is_running:
            self.is_running = True
            self._log("Starting interaction service")
            self.processing_thread = threading.Thread(target=self._process_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            self._log("Processing thread started")
            
    def stop(self):
        """Stop the interaction service."""
        if self.is_running:
            self._log("Stopping interaction service")
            self.is_running = False
            if self.processing_thread:
                self.processing_thread.join()
            self._log("Interaction service stopped")
            
    def set_mode(self, mode):
        """Set the current interaction mode."""
        self._log(f"Setting mode via InteractionService")
        self.manager.set_mode(mode)
