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

from enum import Enum
from typing import Callable, Dict, List, Set

class SpeechEvent(Enum):
    SPEECH_START_POTENTIAL = "speech_start_potential"
    SPEECH_STARTED = "speech_started"
    SPEECH_END_POTENTIAL = "speech_end_potential"
    SPEECH_ENDED = "speech_ended"
    FALSE_START = "false_start"
    FALSE_END = "false_end"
    NEW_TRANSCRIPTION = "new_transcription"

class EventEmitter:
    def __init__(self):
        self._listeners: Dict[SpeechEvent, Set[Callable]] = {event: set() for event in SpeechEvent}

    def on(self, event: SpeechEvent, callback: Callable) -> None:
        """Register a callback for a specific event."""
        self._listeners[event].add(callback)

    def off(self, event: SpeechEvent, callback: Callable) -> None:
        """Remove a callback for a specific event."""
        if callback in self._listeners[event]:
            self._listeners[event].remove(callback)

    def emit(self, event: SpeechEvent, *args, **kwargs) -> None:
        """Emit an event with optional arguments."""
        for callback in self._listeners[event]:
            callback(*args, **kwargs)
