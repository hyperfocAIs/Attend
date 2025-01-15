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
from unittest.mock import Mock, MagicMock
from threading import Lock, Event
from services.interaction.audio import AudioProcessor

@pytest.fixture
def audio_device_manager():
    manager = Mock()
    output_stream = Mock()
    output_stream.is_active.return_value = True
    output_stream.write = Mock()
    manager.create_output_stream.return_value = output_stream
    return manager

@pytest.fixture
def audio_processor(audio_device_manager):
    processor = AudioProcessor(audio_device_manager)
    processor.queue_lock = Lock()
    return processor

@pytest.fixture
def playback_complete():
    return Event()

def test_ensure_output_stream_creates_new_stream(audio_processor, audio_device_manager):
    """Test that _ensure_output_stream creates a new stream when none exists."""
    audio_processor._output_stream = None
    stream = audio_processor._ensure_output_stream()
    
    assert stream is not None
    audio_device_manager.create_output_stream.assert_called_once()

def test_ensure_output_stream_reuses_active_stream(audio_processor, audio_device_manager):
    """Test that _ensure_output_stream reuses an existing active stream."""
    existing_stream = Mock()
    existing_stream.is_active.return_value = True
    audio_processor._output_stream = existing_stream
    
    stream = audio_processor._ensure_output_stream()
    
    assert stream == existing_stream
    audio_device_manager.create_output_stream.assert_not_called()

def test_play_audio_now_writes_chunks(audio_processor, playback_complete):
    """Test that play_audio_now correctly writes audio data in chunks."""
    test_data = b"test" * 1024  # 4KB of test data
    # Set event before playing to avoid initial wait
    playback_complete.set()
    audio_processor.play_audio_now(test_data, playback_complete)
    
    # Verify the stream wrote the data
    stream = audio_processor._output_stream
    total_writes = stream.write.call_count
    assert total_writes > 0

def test_play_audio_now_handles_errors(audio_processor, audio_device_manager, playback_complete):
    """Test that play_audio_now handles errors gracefully."""
    # Ensure we have a stream first
    stream = audio_processor._ensure_output_stream()
    stream.write.side_effect = Exception("Test error")
    
    test_data = b"test" * 1024
    # Set event before playing to avoid initial wait
    playback_complete.set()
    audio_processor.play_audio_now(test_data, playback_complete)
    
    # Verify stream was closed and cleared
    audio_device_manager.close_stream.assert_called_once_with('output')
    assert audio_processor._output_stream is None

def test_queue_audio_adds_to_queue(audio_processor):
    """Test that queue_audio adds audio data to the queue."""
    test_data = b"test audio"
    audio_processor.queue_audio(test_data)
    
    assert len(audio_processor.queued_audio) == 1
    assert audio_processor.queued_audio[0] == test_data

def test_play_queued_audio_plays_all_queued(audio_processor, playback_complete):
    """Test that play_queued_audio plays all queued audio and clears the queue."""
    test_data1 = b"test1"
    test_data2 = b"test2"
    
    audio_processor.queue_audio(test_data1)
    audio_processor.queue_audio(test_data2)
    
    # Set event before playing to avoid initial wait
    playback_complete.set()
    audio_processor.play_queued_audio(playback_complete)
    
    # Verify queue is empty after playing
    assert len(audio_processor.queued_audio) == 0
    
    # Verify stream wrote data
    stream = audio_processor._output_stream
    assert stream.write.call_count > 0

def test_cleanup_closes_stream(audio_processor, audio_device_manager):
    """Test that cleanup properly closes and clears the output stream."""
    # Ensure we have a stream to clean up
    audio_processor._ensure_output_stream()
    
    audio_processor.cleanup()
    
    audio_device_manager.close_stream.assert_called_once_with('output')
    assert audio_processor._output_stream is None
