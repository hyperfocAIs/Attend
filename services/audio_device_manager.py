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

import pyaudio
import threading
import yaml
from typing import Optional, Dict, Tuple

class AudioDeviceManager:
    """
    Centralized manager for audio devices handling both input and output streams.
    Ensures proper resource management and prevents stream conflicts.
    """
    def __init__(self, config_path: str, debug: bool = False):
        """
        Initialize the AudioDeviceManager with configuration from yaml file.
        
        Args:
            config_path: Path to yaml configuration file
            debug: Enable debug mode for additional logging (default: False)
        """
        self._audio = pyaudio.PyAudio()
        self._lock = threading.Lock()
        self._active_streams: Dict[str, pyaudio.Stream] = {}
        self._debug = debug
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self._audio_config = config['client']['audio']
            
        # Store audio parameters from config
        self._rate = self._audio_config['rate']
        self._format = eval(self._audio_config['format'])  # Safely evaluate pyaudio.paInt16
        self._channels = self._audio_config['channels']
        self._chunk = self._audio_config['chunk']
        
        # TTS output configuration (OpenAI TTS format)
        self._output_rate = 24000  # OpenAI TTS outputs at 24kHz
        self._output_format = pyaudio.paInt16  # OpenAI TTS outputs 16-bit PCM
        self._output_channels = 1  # OpenAI TTS outputs mono audio

        # Initialize input/output streams as None
        self._input_stream = None
        self._output_stream = None
        
    @property
    def input_stream(self) -> Optional[pyaudio.Stream]:
        """Get the current input stream."""
        return self._input_stream
        
    @property
    def output_stream(self) -> Optional[pyaudio.Stream]:
        """Get the current output stream."""
        return self._output_stream
        
    def initialize_streams(self, 
                         input_device_index: Optional[int] = None,
                         output_device_index: Optional[int] = None) -> Tuple[pyaudio.Stream, pyaudio.Stream]:
        """
        Initialize both input and output streams using configuration parameters.
        
        Args:
            input_device_index: Specific input device to use (default: None for system default)
            output_device_index: Specific output device to use (default: None for system default)
            
        Returns:
            Tuple of (input_stream, output_stream)
        """
        with self._lock:
            # Create input stream if not exists
            if not self._input_stream or not self._input_stream.is_active():
                self._input_stream = self._audio.open(
                    format=self._format,
                    channels=self._channels,
                    rate=self._rate,
                    input=True,
                    output=False,
                    frames_per_buffer=self._chunk,
                    input_device_index=input_device_index
                )
                self._active_streams['input'] = self._input_stream
                
            # Create output stream if not exists - using TTS output format
            if not self._output_stream or not self._output_stream.is_active():
                self._output_stream = self._audio.open(
                    format=self._output_format,
                    channels=self._output_channels,
                    rate=self._output_rate,
                    input=False,
                    output=True,
                    frames_per_buffer=self._chunk,
                    output_device_index=output_device_index
                )
                self._active_streams['output'] = self._output_stream
                
            return self._input_stream, self._output_stream
            
    def create_input_stream(self, 
                          format: Optional[int] = None,
                          channels: Optional[int] = None,
                          rate: Optional[int] = None,
                          chunk: Optional[int] = None,
                          input_device_index: Optional[int] = None) -> pyaudio.Stream:
        """
        Creates and returns an input stream for recording.
        Uses config values if parameters not specified.
        
        Args:
            format: Audio format (default: from config)
            channels: Number of audio channels (default: from config)
            rate: Sampling rate in Hz (default: from config)
            chunk: Number of frames per buffer (default: from config)
            input_device_index: Specific input device to use (default: None for system default)
            
        Returns:
            PyAudio Stream object configured for input
        """
        with self._lock:
            stream = self._audio.open(
                format=format or self._format,
                channels=channels or self._channels,
                rate=rate or self._rate,
                input=True,
                output=False,
                frames_per_buffer=chunk or self._chunk,
                input_device_index=input_device_index
            )
            self._active_streams['input'] = stream
            return stream
            
    def create_output_stream(self,
                           format: Optional[int] = None,
                           channels: Optional[int] = None,
                           rate: Optional[int] = None,
                           chunk: Optional[int] = None,
                           output_device_index: Optional[int] = None) -> pyaudio.Stream:
        """
        Creates and returns an output stream for playback.
        Uses TTS output format by default.
        
        Args:
            format: Audio format (default: TTS output format)
            channels: Number of audio channels (default: TTS mono)
            rate: Sampling rate in Hz (default: TTS 24kHz)
            chunk: Number of frames per buffer (default: from config)
            output_device_index: Specific output device to use (default: None for system default)
            
        Returns:
            PyAudio Stream object configured for output
        """
        with self._lock:
            stream = self._audio.open(
                format=format or self._output_format,
                channels=channels or self._output_channels,
                rate=rate or self._output_rate,
                input=False,
                output=True,
                frames_per_buffer=chunk or self._chunk,
                output_device_index=output_device_index
            )
            self._active_streams['output'] = stream
            return stream
    
    def close_stream(self, stream_type: str):
        """
        Closes a specific stream (input or output).
        
        Args:
            stream_type: Type of stream to close ('input' or 'output')
        """
        with self._lock:
            if stream_type in self._active_streams:
                stream = self._active_streams[stream_type]
                stream.stop_stream()
                stream.close()
                del self._active_streams[stream_type]
                
                # Clear the corresponding stream reference
                if stream_type == 'input':
                    self._input_stream = None
                elif stream_type == 'output':
                    self._output_stream = None
    
    def close_all_streams(self):
        """Closes all active streams."""
        with self._lock:
            for stream in self._active_streams.values():
                stream.stop_stream()
                stream.close()
            self._active_streams.clear()
            self._input_stream = None
            self._output_stream = None
    
    def terminate(self):
        """Terminates the PyAudio instance and closes all streams."""
        self.close_all_streams()
        self._audio.terminate()
    
    def get_default_input_device_info(self) -> dict:
        """Returns information about the default input device."""
        return self._audio.get_default_input_device_info()
    
    def get_default_output_device_info(self) -> dict:
        """Returns information about the default output device."""
        return self._audio.get_default_output_device_info()
    
    def get_sample_size(self, format: int) -> int:
        """
        Returns the size in bytes of a sample in the specified format.
        
        Args:
            format: PyAudio format constant
            
        Returns:
            Size of a sample in bytes
        """
        return self._audio.get_sample_size(format)
