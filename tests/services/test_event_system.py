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
from services.event_system import EventEmitter, SpeechEvent

@pytest.fixture
def emitter():
    return EventEmitter()

def test_event_registration(emitter):
    """Test that callbacks can be registered for events."""
    called = False
    
    def callback():
        nonlocal called
        called = True
    
    emitter.on(SpeechEvent.SPEECH_STARTED, callback)
    emitter.emit(SpeechEvent.SPEECH_STARTED)
    
    assert called == True

def test_event_removal(emitter):
    """Test that callbacks can be removed from events."""
    called = False
    
    def callback():
        nonlocal called
        called = True
    
    emitter.on(SpeechEvent.SPEECH_STARTED, callback)
    emitter.off(SpeechEvent.SPEECH_STARTED, callback)
    emitter.emit(SpeechEvent.SPEECH_STARTED)
    
    assert called == False

def test_multiple_callbacks(emitter):
    """Test that multiple callbacks can be registered for the same event."""
    call_count = 0
    
    def callback1():
        nonlocal call_count
        call_count += 1
    
    def callback2():
        nonlocal call_count
        call_count += 1
    
    emitter.on(SpeechEvent.SPEECH_STARTED, callback1)
    emitter.on(SpeechEvent.SPEECH_STARTED, callback2)
    emitter.emit(SpeechEvent.SPEECH_STARTED)
    
    assert call_count == 2

def test_event_with_arguments(emitter):
    """Test that events can be emitted with arguments."""
    received_args = None
    received_kwargs = None
    
    def callback(*args, **kwargs):
        nonlocal received_args, received_kwargs
        received_args = args
        received_kwargs = kwargs
    
    emitter.on(SpeechEvent.NEW_TRANSCRIPTION, callback)
    emitter.emit(SpeechEvent.NEW_TRANSCRIPTION, "test text", confidence=0.95)
    
    assert received_args == ("test text",)
    assert received_kwargs == {"confidence": 0.95}

def test_removing_nonexistent_callback(emitter):
    """Test that removing a non-registered callback doesn't raise an error."""
    def callback():
        pass
    
    emitter.off(SpeechEvent.SPEECH_STARTED, callback)  # Should not raise

def test_multiple_events_independence(emitter):
    """Test that different events maintain independent sets of callbacks."""
    start_called = False
    end_called = False
    
    def start_callback():
        nonlocal start_called
        start_called = True
    
    def end_callback():
        nonlocal end_called
        end_called = True
    
    emitter.on(SpeechEvent.SPEECH_STARTED, start_callback)
    emitter.on(SpeechEvent.SPEECH_ENDED, end_callback)
    
    emitter.emit(SpeechEvent.SPEECH_STARTED)
    assert start_called == True
    assert end_called == False
    
    start_called = False  # Reset
    emitter.emit(SpeechEvent.SPEECH_ENDED)
    assert start_called == False
    assert end_called == True
