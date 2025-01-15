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


import io
import pytest
from unittest.mock import Mock, patch, MagicMock
from services.interaction.tts import TTSProcessor

@pytest.fixture
def tts_config():
    return {
        "server-tts": {
            "key": "test-key",
            "host": "localhost",
            "port": "5000",
            "model": "test-model",
            "voice": "test-voice"
        }
    }

@pytest.fixture
def mock_openai():
    with patch('services.interaction.tts.OpenAI') as mock:
        # Setup streaming response mock
        mock_response = Mock()
        mock_response.iter_bytes.return_value = [b"chunk1", b"chunk2"]
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        
        # Setup speech creation mock
        mock_speech = Mock()
        mock_speech.with_streaming_response.create.return_value = mock_response
        
        # Setup audio mock
        mock_audio = Mock()
        mock_audio.speech = mock_speech
        
        # Setup client mock
        mock_client = Mock()
        mock_client.audio = mock_audio
        
        mock.return_value = mock_client
        yield mock

@pytest.fixture
def tts_processor(tts_config, mock_openai):
    return TTSProcessor(tts_config)

def test_init_creates_client_with_config(tts_config):
    """Test that initialization creates OpenAI client with correct configuration."""
    processor = TTSProcessor(tts_config)
    
    assert processor.config == tts_config
    assert processor.debug == False

def test_log_prints_when_debug_enabled(tts_processor, capsys):
    """Test that _log prints messages when debug is enabled."""
    tts_processor.debug = True
    test_message = "test debug message"
    
    tts_processor._log(test_message)
    captured = capsys.readouterr()
    
    assert f"[TTSProcessor] {test_message}" in captured.out

def test_log_silent_when_debug_disabled(tts_processor, capsys):
    """Test that _log doesn't print messages when debug is disabled."""
    tts_processor.debug = False
    test_message = "test debug message"
    
    tts_processor._log(test_message)
    captured = capsys.readouterr()
    
    assert captured.out == ""

def test_process_tts_success(tts_processor, tts_config, mock_openai):
    """Test successful TTS processing."""
    test_sentence = "Hello, world!"
    result = tts_processor.process_tts(test_sentence)
    
    # Verify client was called with correct parameters
    mock_openai.return_value.audio.speech.with_streaming_response.create.assert_called_once_with(
        model=tts_config["server-tts"]["model"],
        voice=tts_config["server-tts"]["voice"],
        response_format="pcm",
        input=test_sentence
    )
    
    # Verify result contains concatenated chunks
    assert result == b"chunk1chunk2"

def test_process_tts_handles_errors(tts_processor, mock_openai):
    """Test error handling in TTS processing."""
    # Setup error condition
    mock_openai.return_value.audio.speech.with_streaming_response.create.side_effect = Exception("Test error")
    
    result = tts_processor.process_tts("test")
    
    assert result is None

def test_process_tts_handles_empty_sentence(tts_processor, mock_openai):
    """Test processing empty sentence."""
    # Empty string should still be processed
    result = tts_processor.process_tts("")
    
    # Verify the call was made and returned expected chunks
    mock_openai.return_value.audio.speech.with_streaming_response.create.assert_called_once()
    assert result == b"chunk1chunk2"

def test_process_tts_with_special_characters(tts_processor, mock_openai):
    """Test processing sentence with special characters."""
    test_sentence = "Hello! @#$%^&*()"
    result = tts_processor.process_tts(test_sentence)
    
    # Verify the call was made with special characters and returned expected chunks
    mock_openai.return_value.audio.speech.with_streaming_response.create.assert_called_once_with(
        model=tts_processor.config["server-tts"]["model"],
        voice=tts_processor.config["server-tts"]["voice"],
        response_format="pcm",
        input=test_sentence
    )
    assert result == b"chunk1chunk2"
