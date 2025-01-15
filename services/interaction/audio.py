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
from typing import List

class AudioProcessor:
    def __init__(self, audio_device_manager):
        self.audio_device_manager = audio_device_manager
        self.queued_audio: List[bytes] = []
        self.queue_lock = None  # Will be set by InteractionManager
        self._output_stream = None

    def _ensure_output_stream(self):
        """Ensure we have an active output stream."""
        if not self._output_stream or not self._output_stream.is_active():
            self._output_stream = self.audio_device_manager.create_output_stream()
        return self._output_stream

    def play_audio_now(self, audio_data: bytes, playback_complete):
        """Play audio immediately."""
        #print("[AudioProcessor] Attempting to play audio now")
        playback_complete.wait()
        playback_complete.clear()
        
        try:
            # Get or create output stream
            output_stream = self._ensure_output_stream()
            #print("[AudioProcessor] Output stream ensured")
            
            # Play the audio
            chunk_size = 1024
            audio_buffer = io.BytesIO(audio_data)
            
            #print("[AudioProcessor] Starting audio playback")
            while True:
                chunk = audio_buffer.read(chunk_size)
                if not chunk:
                    break
                output_stream.write(chunk)
            #print("[AudioProcessor] Audio playback completed")
                
        except Exception as e:
            #print(f"[AudioProcessor] Error playing audio: {e}")
            # If there was an error, close the stream to force recreation next time
            if self._output_stream:
                self.audio_device_manager.close_stream('output')
                self._output_stream = None
        finally:
            playback_complete.set()
            #print("[AudioProcessor] Playback complete event set")

    def queue_audio(self, audio_data: bytes):
        """Queue audio for playback after speech ends."""
        with self.queue_lock:
            self.queued_audio.append(audio_data)
            #print(f"[AudioProcessor] Audio queued. Queue size: {len(self.queued_audio)}")

    def play_queued_audio(self, playback_complete):
        """Play all queued audio."""
        #print("[AudioProcessor] Attempting to play queued audio")
        with self.queue_lock:
            queue_size = len(self.queued_audio)
            #print(f"[AudioProcessor] Queue size: {queue_size}")
            for i, audio_data in enumerate(self.queued_audio, 1):
                #print(f"[AudioProcessor] Playing audio {i} of {queue_size}")
                self.play_audio_now(audio_data, playback_complete)
            self.queued_audio = []
        #print("[AudioProcessor] Finished playing all queued audio")

    def cleanup(self):
        """Clean up resources."""
        if self._output_stream:
            self.audio_device_manager.close_stream('output')
            self._output_stream = None
