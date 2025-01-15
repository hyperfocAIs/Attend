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

import os
import sys
import argparse
from services.audio_device_manager import AudioDeviceManager
from services.manage_recording import AudioRecordingService
from services.interaction.service import InteractionService
import modes.discuss_activities as initial_mode

def main():
    parser = argparse.ArgumentParser(description='Attend - Your AI Assistant')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--config', type=str, default='attend_config.yaml', help='Path to config file')
    args = parser.parse_args()

    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        print("Please copy attend_config-example.yaml to attend_config.yaml and update with your settings")
        sys.exit(1)

    try:
        # Initialize audio device manager with config
        audio_manager = AudioDeviceManager(config_path=args.config, debug=args.debug)
        
        # Initialize audio streams
        try:
            input_stream, output_stream = audio_manager.initialize_streams()
            if args.debug:
                print("Audio streams initialized successfully")
        except Exception as e:
            print(f"Failed to initialize audio streams: {str(e)}")
            sys.exit(1)

        # Initialize recording service with initialized audio manager
        recording_service = AudioRecordingService(
            config_path=args.config,
            debug=args.debug,
            audio_manager=audio_manager
        )

        # Initialize interaction service
        interaction = InteractionService(
            config_path=args.config,
            debug=args.debug,
            recording_service=recording_service,
            audio_device_manager=audio_manager
        )

        # Set initial mode
        interaction.set_mode(initial_mode)

        try:
            # Start services
            recording_service.start()
            if args.debug:
                print("Recording service started")
                
            interaction.start()
            if args.debug:
                print("Interaction service started")

            print("Attend is running. Press Ctrl+C to exit.")
            
            # Keep main thread alive
            while True:
                input()

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            # Stop services in reverse order
            interaction.stop()
            if args.debug:
                print("Interaction service stopped")
                
            recording_service.stop()
            if args.debug:
                print("Recording service stopped")
                
            audio_manager.terminate()
            if args.debug:
                print("Audio manager terminated")

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()