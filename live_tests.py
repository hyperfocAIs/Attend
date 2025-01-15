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


import sys
from tests.usetest.livetest_streamtts import test_stream_text, test_stream_streaming_text
from tests.usetest.livetest_manage_recording import test_recording

def run_live_tests():
    """Run all live tests."""
    print("\n=== Running Live Tests ===\n")
    
    try:
        print("Running TTS Tests...")
        print("-" * 20)
        test_stream_text()
        test_stream_streaming_text()
        print("\nTTS Tests completed successfully")
        
        print("\nRunning Recording Tests...")
        print("-" * 20)
        test_recording()
        print("\nRecording Tests completed successfully")
        
        print("\n=== All Live Tests Completed Successfully ===")
        return 0
        
    except Exception as e:
        print(f"\nError during live tests: {str(e)}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(run_live_tests())
