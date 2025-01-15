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


import time
import yaml
import openai
import nltk
import json
import threading
import queue
import io
import pyaudio
from typing import Optional, Generator

# Download required NLTK data
nltk.download('punkt', quiet=True)

def load_tts_config():
    """Load TTS configuration from attend_config.yaml"""
    with open("attend_config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    
    tts_config = config["server-tts"]
    return {
        'api_key': tts_config["key"],
        'api_base': f"{tts_config['host']}:{tts_config['port']}/v1",
        'model': tts_config["model"],
        'default_voice': tts_config["voice"],
        'default_speed': tts_config["speed"],
        'intersentence_pause': config["client"]["tts"]["intersentence_pause"]
    }

def stream_text(text: str, audio_manager, model: Optional[str] = None, voice: Optional[str] = None, speed: Optional[float] = None) -> dict:
    """
    Stream text-to-speech audio directly to speakers.
    
    Args:
        text (str): The text to convert to speech
        audio_manager: AudioDeviceManager instance for audio output
        voice (str, optional): The voice to use. Defaults to config value.
        speed (float, optional): The speed of the speech. Defaults to config value.
    
    Returns:
        dict: Timing information including time to first byte and total duration
    """
    # Validate output stream
    if not audio_manager.output_stream:
        raise ValueError("AudioDeviceManager must have an initialized output stream")
        
    # Load configuration
    config = load_tts_config()
    
    # Initialize OpenAI client
    client = openai.OpenAI(
        api_key=config['api_key'],
        base_url=config['api_base'],
    )
    
    # Get output stream from audio manager
    player_stream = audio_manager.output_stream
    if not player_stream.is_active():
        raise RuntimeError("Audio output stream is not active")
    
    timing = {}
    start_time = time.time()
    
    try:
        with client.audio.speech.with_streaming_response.create(
            model=model or config['model'],
            voice=voice or config['default_voice'],
            speed=speed or config['default_speed'],
            response_format="pcm",
            input=text,
        ) as response:
            # Record time to first byte
            timing['time_to_first_byte'] = int((time.time() - start_time) * 1000)
            
            # Stream audio chunks
            for chunk in response.iter_bytes(chunk_size=1024):
                if not player_stream.is_active():
                    raise RuntimeError("Audio output stream became inactive")
                player_stream.write(chunk)
            
            # Record total duration
            timing['total_duration'] = int((time.time() - start_time) * 1000)
    
    except Exception as e:
        print(f"Error in TTS streaming: {str(e)}")
        raise
    
    return timing