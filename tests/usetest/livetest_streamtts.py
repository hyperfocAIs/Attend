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
import openai
from functions.streamtts import stream_text, stream_streaming_text
from services.audio_device_manager import AudioDeviceManager

def get_llm_config():
    """Load LLM configuration from attend_config.yaml"""
    with open("attend_config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    return config["server-text"]

def create_streaming_response():
    """Create a streaming chat completion response from the LLM server."""
    config = get_llm_config()
    
    client = openai.OpenAI(
        api_key=config['key'],
        base_url=f"http://{config['host']}:{config['port']}/v1"
    )
    
    return client.chat.completions.create(
        model=config['model'],
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Respond in JSON format with an 'assistant_response' field containing your message."},
            {"role": "user", "content": "Tell me about the benefits of regular exercise in two short sentences."}
        ],
        stream=True
    )

def test_stream_text():
    """Test streaming TTS audio for a simple sentence."""
    # Initialize audio device manager with config
    audio_manager = AudioDeviceManager("attend_config.yaml")
    audio_manager.initialize_streams()
    
    try:
        # Stream a test sentence
        test_sentence = "The quick brown fox jumps over the lazy dog."
        timing = stream_text(test_sentence, audio_manager)
        print(f"TTS Timing: {timing}")
        
    finally:
        # Clean up
        audio_manager.terminate()

def test_stream_streaming_text():
    """Test streaming TTS audio from a live chat completion response."""
    # Initialize audio device manager with config
    audio_manager = AudioDeviceManager("attend_config.yaml")
    audio_manager.initialize_streams()
    
    try:
        # Get streaming response from LLM server
        streaming_response = create_streaming_response()
        
        # Stream the response through TTS
        timing = stream_streaming_text(streaming_response, audio_manager)
        print(f"Streaming TTS Timing: {timing}")
        
    finally:
        # Clean up
        audio_manager.terminate()

if __name__ == "__main__":
    test_stream_text()
    test_stream_streaming_text()
