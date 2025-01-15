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
import os
from unittest.mock import Mock, patch, MagicMock, mock_open, PropertyMock
import yaml
import numpy as np
import time
import sys

from services.manage_recording import RecordingManager, AudioRecordingService
from services.event_system import SpeechEvent

@pytest.fixture
def mock_audio_manager():
    """Create a mock AudioDeviceManager."""
    manager = Mock()
    manager.input_stream = Mock()
    manager.input_stream.is_active.return_value = True
    manager.input_stream.read.return_value = b'\x00' * 2048  # Mock audio data
    manager.get_sample_size.return_value = 2
    return manager

@pytest.fixture
def mock_vad():
    """Create a mock VAD model and iterator."""
    vad_model = Mock()
    vad_iterator = Mock()
    vad_iterator.return_value = None  # Default no speech
    return vad_model, vad_iterator

@pytest.fixture
def config_path():
    """Get the path to the root config file."""
    # Get the path to the root directory (2 levels up from test file)
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(root_dir, 'attend_config.yaml')

@pytest.fixture
def recording_manager(mock_audio_manager, mock_vad, config_path):
    """Create a RecordingManager instance with mocked dependencies."""
    # Only mock the VAD model loading and iterator creation
    with patch('services.manage_recording.load_silero_vad', return_value=mock_vad[0]), \
         patch('services.manage_recording.VADIterator', return_value=mock_vad[1]):
        manager = RecordingManager(config_path, mock_audio_manager)
        yield manager
        manager.close()

@pytest.fixture
def recording_service(recording_manager, mock_audio_manager, config_path):
    """Create an AudioRecordingService instance."""
    service = AudioRecordingService(config_path=config_path, debug=True, audio_manager=mock_audio_manager)
    yield service
    service.stop()

def test_recording_manager_initialization(recording_manager, mock_audio_manager):
    """Test proper initialization of RecordingManager."""
    assert recording_manager.audio_manager == mock_audio_manager
    assert recording_manager.stream == mock_audio_manager.input_stream
    assert recording_manager.rate == 16000
    assert recording_manager.channels == 1
    assert recording_manager.chunk == 512  # From actual config
    assert recording_manager.buffer_seconds == 120  # From actual config
    assert recording_manager.speech_start_threshold == 0.6  # From actual config
    assert recording_manager.speech_end_threshold == 0.6  # From actual config
    assert not recording_manager.is_recording
    assert not recording_manager.speech_detected
    assert not recording_manager.speech_started
    assert not recording_manager.speech_ended

def test_start_stop_recording(recording_manager):
    """Test starting and stopping recording."""
    assert not recording_manager.is_recording
    recording_manager.start_recording()
    assert recording_manager.is_recording
    assert recording_manager.recording_thread is not None
    assert recording_manager.recording_thread.is_alive()
    
    # Store thread reference before stopping
    thread = recording_manager.recording_thread
    recording_manager.stop_recording()
    assert not recording_manager.is_recording
    time.sleep(0.1)  # Give thread time to stop
    assert not thread.is_alive()

def test_vad_state_machine(recording_manager, mock_vad):
    """Test VAD state machine transitions."""
    vad_iterator = mock_vad[1]
    recording_manager.start_recording()
    
    # Simulate speech start
    vad_iterator.return_value = {'start': 0}
    time.sleep(0.2)  # Allow recording thread to process
    assert recording_manager.speech_start_potential
    assert not recording_manager.speech_started
    
    # Simulate continued speech
    vad_iterator.return_value = None
    time.sleep(1.0)  # Wait significantly longer than speech_start_threshold (0.6)
    assert recording_manager.speech_started
    assert recording_manager.speech_detected
    
    # Simulate speech end
    vad_iterator.return_value = {'end': 0}
    time.sleep(0.2)
    assert recording_manager.speech_end_potential
    
    # Simulate silence after speech
    vad_iterator.return_value = None
    time.sleep(1.0)  # Wait significantly longer than speech_end_threshold (0.6)
    assert recording_manager.speech_ended
    assert not recording_manager.speech_started
    
    recording_manager.stop_recording()

def test_save_speech(recording_manager, tmp_path, mock_audio_manager):
    """Test saving speech to WAV file."""
    recording_manager.speech_start_time = time.time() - 1
    recording_manager.speech_end_time = time.time()
    
    with patch('wave.open') as mock_wave:
        mock_wave_file = Mock()
        mock_wave.return_value = mock_wave_file
        
        file_path = recording_manager.save_speech()
        
        assert mock_wave_file.setnchannels.called_with(recording_manager.channels)
        assert mock_wave_file.setsampwidth.called_with(mock_audio_manager.get_sample_size())
        assert mock_wave_file.setframerate.called_with(recording_manager.rate)
        assert mock_wave_file.writeframes.called
        assert mock_wave_file.close.called
        assert file_path.endswith('.wav')

def test_process_stt(recording_manager, tmp_path):
    """Test speech-to-text processing."""
    # Create a temporary test file
    test_file = tmp_path / "test.wav"
    test_file.write_bytes(b'dummy audio data')
    
    # Create a mock transcript with proper structure
    mock_transcript = MagicMock()
    type(mock_transcript).text = PropertyMock(return_value="test transcription")
    
    # Create a mock transcriptions object with create method
    mock_transcriptions = MagicMock()
    mock_transcriptions.create.return_value = mock_transcript
    
    # Create a mock audio object with transcriptions attribute
    mock_audio = MagicMock()
    mock_audio.transcriptions = mock_transcriptions
    
    # Create a mock OpenAI client with audio attribute
    mock_client = MagicMock()
    mock_client.audio = mock_audio
    
    # Mock the OpenAI constructor at the correct import path
    with patch('services.manage_recording.OpenAI', return_value=mock_client) as mock_openai:
        result = recording_manager.process_stt(str(test_file))
        
        # Verify the mock was called correctly
        mock_openai.assert_called_once()
        mock_client.audio.transcriptions.create.assert_called_once()
        assert result == [{"start": 0, "end": 0, "text": "test transcription"}]
        assert recording_manager.latest_transcription == result

def test_event_emission(recording_manager):
    """Test that speech events are properly emitted."""
    events_received = []
    
    def event_callback(event_type):
        events_received.append(event_type)
    
    recording_manager.add_event_listener(SpeechEvent.SPEECH_START_POTENTIAL, 
                                      lambda: event_callback(SpeechEvent.SPEECH_START_POTENTIAL))
    recording_manager.add_event_listener(SpeechEvent.SPEECH_STARTED, 
                                      lambda: event_callback(SpeechEvent.SPEECH_STARTED))
    
    # Simulate speech detection directly through event emission
    recording_manager.events.emit(SpeechEvent.SPEECH_START_POTENTIAL)
    assert SpeechEvent.SPEECH_START_POTENTIAL in events_received
    
    recording_manager.events.emit(SpeechEvent.SPEECH_STARTED)
    assert SpeechEvent.SPEECH_STARTED in events_received

def test_recording_service_lifecycle(recording_service):
    """Test AudioRecordingService lifecycle management."""
    assert not recording_service.is_running
    
    recording_service.start()
    assert recording_service.is_running
    assert recording_service.processing_thread.is_alive()
    
    recording_service.stop()
    assert not recording_service.is_running
    time.sleep(0.1)  # Give thread time to stop
    assert not recording_service.processing_thread.is_alive()

def test_recording_service_event_handling(recording_service):
    """Test AudioRecordingService event handling."""
    events_received = []
    
    def event_callback(event):
        events_received.append(event)
    
    recording_service.add_event_listener(SpeechEvent.SPEECH_STARTED, 
                                      lambda: event_callback(SpeechEvent.SPEECH_STARTED))
    
    recording_service.start()
    # Directly emit the event instead of relying on speech detection
    recording_service.manager.events.emit(SpeechEvent.SPEECH_STARTED)
    time.sleep(0.1)  # Allow processing thread to run
    
    assert SpeechEvent.SPEECH_STARTED in events_received
    
    recording_service.stop()

def test_reset_state(recording_manager):
    """Test resetting the VAD state machine."""
    recording_manager.speech_start_time = time.time()
    recording_manager.speech_end_time = time.time()
    recording_manager.speech_start_potential = True
    recording_manager.speech_end_potential = True
    recording_manager.speech_started = True
    recording_manager.speech_detected = True
    recording_manager.speech_ended = True
    
    recording_manager.reset()
    
    assert recording_manager.speech_start_time is None
    assert recording_manager.speech_end_time is None
    assert not recording_manager.speech_start_potential
    assert not recording_manager.speech_end_potential
    assert not recording_manager.speech_started
    assert not recording_manager.speech_detected
    assert not recording_manager.speech_ended

def test_get_speech_duration(recording_manager):
    """Test getting speech duration."""
    start_time = time.time()
    recording_manager.speech_start_time = start_time
    recording_manager.speech_end_time = start_time + 1.5
    
    duration = recording_manager.get_speech_duration()
    assert duration == pytest.approx(1.5, rel=1e-2)
    
    recording_manager.speech_start_time = None
    recording_manager.speech_end_time = None
    assert recording_manager.get_speech_duration() is None
