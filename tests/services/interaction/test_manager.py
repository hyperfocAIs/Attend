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
import threading
from unittest.mock import Mock, patch, MagicMock
from services.interaction.manager import InteractionManager
from services.manage_recording import AudioRecordingService
from services.event_system import EventEmitter, SpeechEvent

@pytest.fixture
def config():
    return {
        "server-text": {
            "key": "test-key",
            "host": "localhost",
            "port": "5000",
            "model": "test-model"
        },
        "server-tts": {
            "key": "test-key",
            "host": "localhost",
            "port": "5000",
            "model": "test-model",
            "voice": "test-voice"
        }
    }

@pytest.fixture
def mock_config_path(tmp_path, config):
    """Create a temporary config file."""
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    return str(config_file)

@pytest.fixture
def mock_recording_service():
    return Mock(spec=AudioRecordingService)

@pytest.fixture
def mock_audio_device_manager():
    return Mock()

@pytest.fixture
def mock_openai():
    with patch('services.interaction.manager.OpenAI') as mock:
        mock_client = Mock()
        mock_chat = Mock()
        mock_completions = Mock()
        mock_client.chat = mock_chat
        mock_chat.completions = mock_completions
        mock.return_value = mock_client
        yield mock

@pytest.fixture
def interaction_manager(mock_config_path, mock_recording_service, mock_audio_device_manager, mock_openai):
    with patch('services.interaction.manager.AudioProcessor') as mock_audio_processor, \
         patch('services.interaction.manager.TTSProcessor') as mock_tts_processor:
        manager = InteractionManager(
            mock_config_path,
            mock_recording_service,
            mock_audio_device_manager
        )
        yield manager

def test_init_creates_manager_with_config(interaction_manager, config):
    """Test that initialization creates manager with correct configuration."""
    assert interaction_manager.config == config
    assert interaction_manager.debug == False
    assert isinstance(interaction_manager.events, EventEmitter)
    assert interaction_manager.current_mode is None
    assert interaction_manager.messages == []
    assert interaction_manager.messages_tentative == []

def test_log_prints_when_debug_enabled(interaction_manager, capsys):
    """Test that _log prints messages when debug is enabled."""
    interaction_manager.debug = True
    test_message = "test debug message"
    
    interaction_manager._log(test_message)
    captured = capsys.readouterr()
    
    assert f"[InteractionManager] {test_message}" in captured.out

def test_log_silent_when_debug_disabled(interaction_manager, capsys):
    """Test that _log doesn't print messages when debug is disabled."""
    interaction_manager.debug = False
    test_message = "test debug message"
    
    interaction_manager._log(test_message)
    captured = capsys.readouterr()
    
    assert captured.out == ""

def test_handle_speech_end_potential(interaction_manager):
    """Test handling of potential speech end."""
    interaction_manager._handle_speech_end_potential()
    
    assert interaction_manager.in_potential_end_state == True

def test_handle_false_end(interaction_manager):
    """Test handling of false end detection."""
    # Setup initial state
    interaction_manager.in_potential_end_state = True
    interaction_manager.current_pipeline = "test_pipeline"
    interaction_manager.audio_processor.queued_audio = ["test_audio"]
    
    interaction_manager._handle_false_end()
    
    assert interaction_manager.in_potential_end_state == False
    assert interaction_manager.current_pipeline is None
    assert interaction_manager.audio_processor.queued_audio == []

def test_handle_speech_ended(interaction_manager):
    """Test handling of confirmed speech end."""
    interaction_manager._handle_speech_ended()
    
    assert interaction_manager.speech_ended.is_set()
    interaction_manager.audio_processor.play_queued_audio.assert_called_once_with(
        interaction_manager.playback_complete
    )

def test_handle_transcription_empty(interaction_manager):
    """Test handling of empty transcription."""
    interaction_manager._handle_transcription([])
    
    assert interaction_manager.messages == []
    assert interaction_manager.messages_tentative == []

def test_handle_transcription_success(interaction_manager, mock_openai):
    """Test successful transcription handling."""
    # Mock LLM response
    mock_chunk = Mock()
    mock_choice = Mock()
    mock_delta = Mock()
    mock_delta.content = '{"assistant_response": "Hello!"}'
    mock_choice.delta = mock_delta
    mock_chunk.choices = [mock_choice]
    
    mock_openai.return_value.chat.completions.create.return_value = [mock_chunk]
    
    # Test transcription handling
    interaction_manager._handle_transcription([{"text": "Hello"}])
    
    # Verify LLM was called
    mock_openai.return_value.chat.completions.create.assert_called_once()
    
    # Verify messages were updated
    assert len(interaction_manager.messages) == 2
    assert interaction_manager.messages[0]["role"] == "user"
    assert interaction_manager.messages[0]["content"] == "Hello"
    assert interaction_manager.messages[1]["role"] == "assistant"
    assert interaction_manager.messages[1]["content"] == '{"assistant_response": "Hello!"}'

def test_set_mode_basic(interaction_manager):
    """Test setting mode without initialization."""
    mock_mode = Mock()
    
    interaction_manager.set_mode(mock_mode)
    
    assert interaction_manager.current_mode == mock_mode

def test_set_mode_with_initialization(interaction_manager):
    """Test setting mode with initialization configuration."""
    mock_mode = Mock()
    mock_mode.initialize = {
        "greeting": {
            "text": "Hello!",
            "speed": 1.0
        }
    }
    mock_mode.system_prompt = "System prompt"
    
    with patch('services.interaction.manager.stream_text') as mock_stream:
        interaction_manager.set_mode(mock_mode)
        
        mock_stream.assert_called_once_with(
            text="Hello!",
            audio_manager=interaction_manager.audio_device_manager,
            speed=1.0
        )
        
        assert interaction_manager.messages == [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Let's get to it."},
            {"role": "assistant", "content": "Hello!"}
        ]

def test_set_mode_with_lifecycle_methods(interaction_manager):
    """Test setting mode with lifecycle methods."""
    mock_mode = Mock()
    mock_mode.before_first_turn = Mock()
    mock_mode.after_attend_turn = Mock()
    
    interaction_manager.set_mode(mock_mode)
    
    mock_mode.before_first_turn.assert_called_once()
    mock_mode.after_attend_turn.assert_called_once()

def test_set_mode_with_prompt_initialization(interaction_manager):
    """Test setting mode with prompt initialization."""
    mock_mode = Mock()
    mock_mode.initialize = {
        "prompt": "Test prompt"
    }
    
    interaction_manager.set_mode(mock_mode)
    
    # Currently prompt initialization is not implemented
    assert interaction_manager.current_mode == mock_mode
    assert interaction_manager.messages_tentative == []

def test_handle_transcription_error(interaction_manager, mock_openai):
    """Test error handling in transcription processing."""
    # Set up initial state
    initial_messages = [{"role": "system", "content": "Initial message"}]
    interaction_manager.messages = initial_messages.copy()
    interaction_manager.messages_tentative = initial_messages.copy()
    
    # Setup error condition
    mock_openai.return_value.chat.completions.create.side_effect = Exception("Test error")
    
    # Should not raise exception
    interaction_manager._handle_transcription([{"text": "Hello"}])
    
    # Verify only user message was added to tentative messages
    expected_tentative = initial_messages + [{"role": "user", "content": "Hello"}]
    assert interaction_manager.messages_tentative == expected_tentative
    
    # Verify main messages remain unchanged
    assert interaction_manager.messages == initial_messages

def test_handle_transcription_pipeline_abort(interaction_manager, mock_openai):
    """Test pipeline abort during transcription processing."""
    # Set up initial state
    initial_messages = [{"role": "system", "content": "Initial message"}]
    interaction_manager.messages = initial_messages.copy()
    interaction_manager.messages_tentative = initial_messages.copy()
    
    # Setup mock response
    mock_chunk = Mock()
    mock_choice = Mock()
    mock_delta = Mock()
    mock_delta.content = '{"assistant_response": "Hello!"}'
    mock_choice.delta = mock_delta
    mock_chunk.choices = [mock_choice]
    
    # Process transcription with aborted pipeline
    interaction_manager.current_pipeline = None
    interaction_manager._handle_transcription([{"text": "Hello"}])
    
    # Verify only user message was added to tentative messages
    expected_tentative = initial_messages + [{"role": "user", "content": "Hello"}]
    assert interaction_manager.messages_tentative == expected_tentative
    
    # Verify main messages remain unchanged
    assert interaction_manager.messages == initial_messages

def test_handle_transcription_json_error(interaction_manager, mock_openai):
    """Test handling of invalid JSON in transcription response."""
    # Set up initial state
    initial_messages = [{"role": "system", "content": "Initial message"}]
    interaction_manager.messages = initial_messages.copy()
    interaction_manager.messages_tentative = initial_messages.copy()
    
    # Mock LLM response with invalid JSON
    mock_chunk = Mock()
    mock_choice = Mock()
    mock_delta = Mock()
    mock_delta.content = 'invalid json'
    mock_choice.delta = mock_delta
    mock_chunk.choices = [mock_choice]
    
    mock_openai.return_value.chat.completions.create.return_value = [mock_chunk]
    
    # Process transcription
    interaction_manager._handle_transcription([{"text": "Hello"}])
    
    # Verify only user message was added to tentative messages
    expected_tentative = initial_messages + [{"role": "user", "content": "Hello"}]
    assert interaction_manager.messages_tentative == expected_tentative
    
    # Verify main messages remain unchanged
    assert interaction_manager.messages == initial_messages
    
    # Verify no assistant message was added
    assert not any(msg["role"] == "assistant" for msg in interaction_manager.messages_tentative)

def test_handle_transcription_sentence_processing(interaction_manager, mock_openai):
    """Test sentence processing in transcription handling."""
    # Mock LLM response with multiple sentences
    mock_chunk = Mock()
    mock_choice = Mock()
    mock_delta = Mock()
    mock_delta.content = '{"assistant_response": "Hello! How are you?"}'
    mock_choice.delta = mock_delta
    mock_chunk.choices = [mock_choice]
    
    mock_openai.return_value.chat.completions.create.return_value = [mock_chunk]
    
    with patch('nltk.sent_tokenize') as mock_tokenize:
        mock_tokenize.return_value = ["Hello!", "How are you?"]
        
        interaction_manager._handle_transcription([{"text": "Hi"}])
        
        # Verify TTS was called for each sentence
        assert interaction_manager.tts_processor.process_tts.call_count == 2
        interaction_manager.tts_processor.process_tts.assert_any_call("Hello!")
        interaction_manager.tts_processor.process_tts.assert_any_call("How are you?")

def test_event_registration(mock_config_path, mock_recording_service, mock_audio_device_manager, mock_openai):
    """Test event registration during initialization."""
    with patch('services.interaction.manager.AudioProcessor') as mock_audio_processor, \
         patch('services.interaction.manager.TTSProcessor') as mock_tts_processor, \
         patch('services.interaction.manager.EventEmitter') as mock_event_emitter:
        
        manager = InteractionManager(
            mock_config_path,
            mock_recording_service,
            mock_audio_device_manager
        )
        
        # Verify event handlers were registered
        mock_event_emitter.return_value.on.assert_any_call(
            SpeechEvent.SPEECH_END_POTENTIAL,
            manager._handle_speech_end_potential
        )
        mock_event_emitter.return_value.on.assert_any_call(
            SpeechEvent.FALSE_END,
            manager._handle_false_end
        )
        mock_event_emitter.return_value.on.assert_any_call(
            SpeechEvent.SPEECH_ENDED,
            manager._handle_speech_ended
        )
        mock_event_emitter.return_value.on.assert_any_call(
            SpeechEvent.NEW_TRANSCRIPTION,
            manager._handle_transcription
        )
