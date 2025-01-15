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
from unittest.mock import Mock, patch
from services.interaction.service import InteractionService
from services.manage_recording import AudioRecordingService
from services.event_system import SpeechEvent

@pytest.fixture
def mock_audio_recording_service():
    return Mock(spec=AudioRecordingService)

@pytest.fixture
def mock_audio_device_manager():
    return Mock()

@pytest.fixture
def mock_interaction_manager():
    with patch('services.interaction.service.InteractionManager') as mock:
        yield mock.return_value

@pytest.fixture
def interaction_service(mock_audio_recording_service, mock_audio_device_manager, mock_interaction_manager):
    return InteractionService(
        config_path="test_config.yaml",
        debug=True,
        recording_service=mock_audio_recording_service,
        audio_device_manager=mock_audio_device_manager
    )

def test_interaction_service_initialization(interaction_service, mock_audio_recording_service):
    assert interaction_service.debug == True
    assert interaction_service.recording_service == mock_audio_recording_service
    assert interaction_service.is_running == False
    assert interaction_service.processing_thread is None

    # Check if event listeners are set up correctly
    mock_audio_recording_service.add_event_listener.assert_any_call(
        SpeechEvent.NEW_TRANSCRIPTION, 
        interaction_service.manager._handle_transcription
    )
    mock_audio_recording_service.add_event_listener.assert_any_call(
        SpeechEvent.SPEECH_END_POTENTIAL,
        interaction_service.manager._handle_speech_end_potential
    )
    mock_audio_recording_service.add_event_listener.assert_any_call(
        SpeechEvent.FALSE_END,
        interaction_service.manager._handle_false_end
    )
    mock_audio_recording_service.add_event_listener.assert_any_call(
        SpeechEvent.SPEECH_ENDED,
        interaction_service.manager._handle_speech_ended
    )

def test_interaction_service_initialization_without_recording_service():
    with pytest.raises(ValueError, match="AudioRecordingService instance must be provided"):
        InteractionService(config_path="test_config.yaml", audio_device_manager=Mock())

def test_interaction_service_initialization_without_audio_device_manager():
    with pytest.raises(ValueError, match="AudioDeviceManager instance must be provided"):
        InteractionService(config_path="test_config.yaml", recording_service=Mock())

def test_start_and_stop(interaction_service):
    interaction_service.start()
    assert interaction_service.is_running == True
    assert interaction_service.processing_thread is not None
    assert interaction_service.processing_thread.is_alive()

    interaction_service.stop()
    assert interaction_service.is_running == False
    assert not interaction_service.processing_thread.is_alive()

def test_set_mode(interaction_service, mock_interaction_manager):
    test_mode = "test_mode"
    interaction_service.set_mode(test_mode)
    mock_interaction_manager.set_mode.assert_called_once_with(test_mode)

def test_log_debug_enabled(interaction_service, capsys):
    interaction_service._log("Test message")
    captured = capsys.readouterr()
    assert "[InteractionService] Test message" in captured.out

def test_log_debug_disabled(interaction_service, capsys):
    interaction_service.debug = False
    interaction_service._log("Test message")
    captured = capsys.readouterr()
    assert captured.out == ""

@patch('time.sleep')
def test_process_loop(mock_sleep, interaction_service):
    interaction_service.is_running = True
    
    def stop_after_two_iterations(sleep_time):
        nonlocal iteration_count
        iteration_count += 1
        if iteration_count >= 2:
            interaction_service.is_running = False
        assert sleep_time == 0.01

    iteration_count = 0
    mock_sleep.side_effect = stop_after_two_iterations

    interaction_service._process_loop()

    assert mock_sleep.call_count == 2

@patch('time.sleep', side_effect=Exception("Test exception"))
def test_process_loop_exception_handling(mock_sleep, interaction_service):
    interaction_service.is_running = True
    interaction_service._process_loop()
    assert interaction_service.is_running == False

def test_start_when_already_running(interaction_service):
    interaction_service.start()
    assert interaction_service.is_running == True
    assert interaction_service.processing_thread is not None
    
    # Try to start again
    initial_thread = interaction_service.processing_thread
    interaction_service.start()
    
    # Verify that a new thread wasn't created
    assert interaction_service.processing_thread == initial_thread

def test_stop_when_not_running(interaction_service):
    assert interaction_service.is_running == False
    assert interaction_service.processing_thread is None
    
    # Try to stop when not running
    interaction_service.stop()
    
    # Verify that the state hasn't changed
    assert interaction_service.is_running == False
    assert interaction_service.processing_thread is None

@patch('threading.Thread')
def test_start_creates_daemon_thread(mock_thread, interaction_service):
    interaction_service.start()
    
    mock_thread.assert_called_once()
    thread_instance = mock_thread.return_value
    assert thread_instance.daemon == True
    thread_instance.start.assert_called_once()
