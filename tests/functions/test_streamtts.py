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



import pytest
import yaml
import json
import threading
import queue
from unittest.mock import Mock, patch, MagicMock, mock_open
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from functions.streamtts import load_tts_config, stream_text, stream_streaming_text

@pytest.fixture
def mock_config():
    """Create a mock TTS configuration."""
    return {
        "server-tts": {
            "key": "test-key",
            "host": "localhost",
            "port": "8000",
            "model": "test-model",
            "voice": "test-voice",
            "speed": 1.0
        },
        "client": {
            "tts": {
                "intersentence_pause": 0.1
            }
        }
    }

@pytest.fixture
def mock_audio_manager():
    """Create a mock AudioDeviceManager."""
    manager = Mock()
    manager.output_stream = Mock()
    manager.output_stream.is_active.return_value = True
    manager.output_stream.write = Mock()
    return manager

@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI streaming response."""
    response = MagicMock()
    response.iter_bytes.return_value = [b'chunk1', b'chunk2']
    response.__enter__.return_value = response
    response.__exit__.return_value = None
    return response

@pytest.fixture
def mock_streaming_response():
    """Create a mock streaming response from OpenAI chat completions."""
    class MockDelta:
        def __init__(self, content):
            self.content = content

    class MockChoice:
        def __init__(self, content):
            self.delta = MockDelta(content)

    class MockChunk:
        def __init__(self, content):
            self.choices = [MockChoice(content)]

    # Create response chunks that build a valid JSON string
    return [
        MockChunk('{"assistant_response": "This is sentence one."}'),
        MockChunk('{"assistant_response": "This is sentence one. This is sentence two."}')
    ]

def test_load_tts_config(mock_config):
    """Test loading TTS configuration from file."""
    mock_file = mock_open(read_data=yaml.dump(mock_config))
    
    with patch('builtins.open', mock_file):
        config = load_tts_config()
        
        assert config['api_key'] == 'test-key'
        assert config['api_base'] == 'http://localhost:8000/v1'
        assert config['model'] == 'test-model'
        assert config['default_voice'] == 'test-voice'
        assert config['default_speed'] == 1.0
        assert config['intersentence_pause'] == 0.1

def test_stream_text_success(mock_audio_manager, mock_openai_response, mock_config):
    """Test successful text-to-speech streaming."""
    with patch('builtins.open', mock_open(read_data=yaml.dump(mock_config))), \
         patch('openai.OpenAI') as mock_openai:
        
        # Configure mock OpenAI client
        mock_client = Mock()
        mock_client.audio.speech.with_streaming_response.create.return_value = mock_openai_response
        mock_openai.return_value = mock_client
        
        # Call stream_text
        timing = stream_text(
            "Test text",
            mock_audio_manager,
            model="test-model",
            voice="test-voice",
            speed=1.0
        )
        
        # Verify OpenAI client was configured correctly
        mock_openai.assert_called_once_with(
            api_key='test-key',
            base_url='http://localhost:8000/v1'
        )
        
        # Verify audio streaming
        assert mock_audio_manager.output_stream.write.call_count == 2
        assert 'time_to_first_byte' in timing
        assert 'total_duration' in timing

def test_stream_text_inactive_stream(mock_audio_manager, mock_config):
    """Test stream_text with inactive audio stream."""
    mock_audio_manager.output_stream.is_active.return_value = False
    
    with patch('builtins.open', mock_open(read_data=yaml.dump(mock_config))):
        with pytest.raises(RuntimeError, match="Audio output stream is not active"):
            stream_text("Test text", mock_audio_manager)

def test_stream_text_no_output_stream(mock_config):
    """Test stream_text with no output stream configured."""
    mock_audio_manager = Mock()
    mock_audio_manager.output_stream = None
    
    with patch('builtins.open', mock_open(read_data=yaml.dump(mock_config))):
        with pytest.raises(ValueError, match="AudioDeviceManager must have an initialized output stream"):
            stream_text("Test text", mock_audio_manager)

def test_stream_streaming_text_success(mock_audio_manager, mock_streaming_response, mock_config, mock_openai_response):
    """Test successful streaming text-to-speech from chat completion."""
    with patch('builtins.open', mock_open(read_data=yaml.dump(mock_config))), \
         patch('openai.OpenAI') as mock_openai, \
         patch('time.sleep'), \
         patch('nltk.sent_tokenize', return_value=["This is sentence one.", "This is sentence two."]), \
         patch('queue.Queue') as mock_queue_class:
        
        # Configure mock queues
        mock_sentence_queue = MagicMock()
        mock_audio_queue = MagicMock()
        mock_queue_class.side_effect = [mock_sentence_queue, mock_audio_queue]
        
        # Configure mock OpenAI client
        mock_client = Mock()
        mock_client.audio.speech.with_streaming_response.create.return_value = mock_openai_response
        mock_openai.return_value = mock_client
        
        # Call stream_streaming_text
        timing = stream_streaming_text(
            mock_streaming_response,
            mock_audio_manager,
            model="test-model",
            voice="test-voice",
            speed=1.0
        )
        
        # Verify OpenAI client was configured correctly
        mock_openai.assert_called_with(
            api_key='test-key',
            base_url='http://localhost:8000/v1'
        )
        
        # Verify timing information is returned
        assert 'time_to_first_byte' in timing
        assert 'total_duration' in timing

def test_stream_streaming_text_inactive_stream(mock_audio_manager, mock_streaming_response, mock_config):
    """Test stream_streaming_text with inactive audio stream."""
    mock_audio_manager.output_stream.is_active.return_value = False
    
    with patch('builtins.open', mock_open(read_data=yaml.dump(mock_config))):
        with pytest.raises(RuntimeError, match="Audio output stream is not active"):
            stream_streaming_text(mock_streaming_response, mock_audio_manager)

def test_stream_streaming_text_no_output_stream(mock_streaming_response, mock_config):
    """Test stream_streaming_text with no output stream configured."""
    mock_audio_manager = Mock()
    mock_audio_manager.output_stream = None
    
    with patch('builtins.open', mock_open(read_data=yaml.dump(mock_config))):
        with pytest.raises(ValueError, match="AudioDeviceManager must have an initialized output stream"):
            stream_streaming_text(mock_streaming_response, mock_audio_manager)

def test_stream_streaming_text_invalid_json(mock_audio_manager, mock_config, mock_openai_response):
    """Test stream_streaming_text with invalid JSON in response."""
    class MockDelta:
        def __init__(self, content):
            self.content = content

    class MockChoice:
        def __init__(self, content):
            self.delta = MockDelta(content)

    class MockChunk:
        def __init__(self, content):
            self.choices = [MockChoice(content)]

    # Create response with invalid JSON that completes immediately
    invalid_json_response = [
        MockChunk('{"invalid json'),
        MockChunk('"}')  # Complete the response to avoid hanging
    ]
    
    with patch('builtins.open', mock_open(read_data=yaml.dump(mock_config))), \
         patch('openai.OpenAI') as mock_openai, \
         patch('time.sleep'), \
         patch('nltk.sent_tokenize', return_value=[]), \
         patch('queue.Queue') as mock_queue_class:
        
        # Configure mock queues that complete immediately
        mock_sentence_queue = queue.Queue()
        mock_audio_queue = queue.Queue()
        mock_queue_class.side_effect = [mock_sentence_queue, mock_audio_queue]
        
        # Configure mock OpenAI client
        mock_client = Mock()
        mock_client.audio.speech.with_streaming_response.create.return_value = mock_openai_response
        mock_openai.return_value = mock_client
        
        # The function should handle invalid JSON gracefully
        timing = stream_streaming_text(
            invalid_json_response,
            mock_audio_manager
        )
        
        # Verify timing information is returned
        assert 'time_to_first_byte' in timing
        assert 'total_duration' in timing

def test_stream_streaming_text_tts_error(mock_audio_manager, mock_streaming_response, mock_config):
    """Test stream_streaming_text handling TTS API errors."""
    with patch('builtins.open', mock_open(read_data=yaml.dump(mock_config))), \
         patch('openai.OpenAI') as mock_openai, \
         patch('time.sleep'), \
         patch('nltk.sent_tokenize', return_value=["This is sentence one.", "This is sentence two."]), \
         patch('queue.Queue') as mock_queue_class:
        
        # Configure mock queues that complete immediately
        mock_sentence_queue = queue.Queue()
        mock_audio_queue = queue.Queue()
        mock_queue_class.side_effect = [mock_sentence_queue, mock_audio_queue]
        
        # Configure mock OpenAI client to raise an exception
        mock_client = Mock()
        mock_client.audio.speech.with_streaming_response.create.side_effect = Exception("TTS API Error")
        mock_openai.return_value = mock_client
        
        # The function should handle TTS errors gracefully
        timing = stream_streaming_text(
            mock_streaming_response,
            mock_audio_manager
        )
        
        # Verify timing information is returned
        assert 'time_to_first_byte' in timing
        assert 'total_duration' in timing
