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
from openai import OpenAI

class TTSProcessor:
    def __init__(self, config):
        """Initialize TTS processor with configuration."""
        tts_config = config["server-tts"]
        self.client = OpenAI(
            api_key=tts_config["key"],
            base_url=f"{tts_config['host']}:{tts_config['port']}/v1"
        )
        self.config = config
        self.debug = False

    def _log(self, message):
        """Print debug messages if debug mode is enabled."""
        if self.debug:
            print(f"[TTSProcessor] {message}")

    def process_tts(self, sentence: str) -> bytes:
        """Process a sentence through TTS and return the audio data."""
        self._log(f"Processing TTS for sentence: {sentence}")
        try:
            audio_data = io.BytesIO()
            with self.client.audio.speech.with_streaming_response.create(
                model=self.config["server-tts"]["model"],
                voice=self.config["server-tts"]["voice"],
                response_format="pcm",
                input=sentence,
            ) as response:
                for chunk in response.iter_bytes(chunk_size=1024):
                    audio_data.write(chunk)
            self._log("TTS processing completed successfully")
            return audio_data.getvalue()
        except Exception as e:
            self._log(f"Error in TTS processing: {e}")
            return None
