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

import unittest
from unittest.mock import Mock, patch, mock_open, MagicMock
import sys

# Create a mock pyaudio module
mock_pyaudio = MagicMock()
mock_pyaudio.paInt16 = 8
mock_pyaudio.paFloat32 = 32
sys.modules['pyaudio'] = mock_pyaudio

import yaml
from services.audio_device_manager import AudioDeviceManager

# Mock configuration for testing
MOCK_CONFIG = """
client:
    audio:
        rate: 16000
        format: pyaudio.paInt16
        channels: 1
        chunk: 1024
"""

class TestAudioDeviceManager(unittest.TestCase):
    def setUp(self):
        # Create a mock PyAudio instance
        self.mock_pyaudio = Mock()
        self.mock_stream = Mock()
        
        # Configure mock stream
        self.mock_stream.is_active.return_value = True
        self.mock_stream.stop_stream = Mock()
        self.mock_stream.close = Mock()
        
        # Configure PyAudio mock to return our mock stream
        self.mock_pyaudio.open.return_value = self.mock_stream
        
        # Mock device info
        self.mock_device_info = {
            'index': 0,
            'name': 'Mock Device',
            'maxInputChannels': 2,
            'maxOutputChannels': 2,
            'defaultSampleRate': 44100
        }
        self.mock_pyaudio.get_default_input_device_info.return_value = self.mock_device_info
        self.mock_pyaudio.get_default_output_device_info.return_value = self.mock_device_info
        self.mock_pyaudio.get_sample_size = Mock(return_value=2)
        
        # Mock the PyAudio class itself
        self.pyaudio_patcher = patch('pyaudio.PyAudio', return_value=self.mock_pyaudio)
        self.pyaudio_patcher.start()
        
        # Create manager instance with mock config
        with patch('builtins.open', mock_open(read_data=MOCK_CONFIG)):
            self.manager = AudioDeviceManager('mock_config.yaml')

    def tearDown(self):
        self.pyaudio_patcher.stop()

    def test_initialization(self):
        """Test proper initialization of AudioDeviceManager"""
        self.assertIsNone(self.manager.input_stream)
        self.assertIsNone(self.manager.output_stream)
        self.assertEqual(self.manager._rate, 16000)
        self.assertEqual(self.manager._channels, 1)
        self.assertEqual(self.manager._chunk, 1024)

    def test_initialize_streams(self):
        """Test initialization of both input and output streams"""
        input_stream, output_stream = self.manager.initialize_streams()
        
        # Verify streams were created
        self.assertIsNotNone(input_stream)
        self.assertIsNotNone(output_stream)
        
        # Verify PyAudio.open was called twice (once for each stream)
        self.assertEqual(self.mock_pyaudio.open.call_count, 2)
        
        # Verify streams were stored in active_streams
        self.assertIn('input', self.manager._active_streams)
        self.assertIn('output', self.manager._active_streams)

    def test_create_input_stream(self):
        """Test creation of input stream with custom parameters"""
        custom_format = mock_pyaudio.paFloat32
        custom_channels = 2
        custom_rate = 44100
        custom_chunk = 2048
        
        stream = self.manager.create_input_stream(
            format=custom_format,
            channels=custom_channels,
            rate=custom_rate,
            chunk=custom_chunk
        )
        
        # Verify stream was created with custom parameters
        self.mock_pyaudio.open.assert_called_with(
            format=custom_format,
            channels=custom_channels,
            rate=custom_rate,
            input=True,
            output=False,
            frames_per_buffer=custom_chunk,
            input_device_index=None
        )
        
        self.assertIs(stream, self.mock_stream)
        self.assertIn('input', self.manager._active_streams)

    def test_create_output_stream(self):
        """Test creation of output stream with custom parameters"""
        custom_format = mock_pyaudio.paFloat32
        custom_channels = 2
        custom_rate = 44100
        custom_chunk = 2048
        
        stream = self.manager.create_output_stream(
            format=custom_format,
            channels=custom_channels,
            rate=custom_rate,
            chunk=custom_chunk
        )
        
        # Verify stream was created with custom parameters
        self.mock_pyaudio.open.assert_called_with(
            format=custom_format,
            channels=custom_channels,
            rate=custom_rate,
            input=False,
            output=True,
            frames_per_buffer=custom_chunk,
            output_device_index=None
        )
        
        self.assertIs(stream, self.mock_stream)
        self.assertIn('output', self.manager._active_streams)

    def test_close_stream(self):
        """Test closing a specific stream"""
        # First create a stream
        self.manager.create_input_stream()
        
        # Then close it
        self.manager.close_stream('input')
        
        # Verify stream was properly closed
        self.mock_stream.stop_stream.assert_called_once()
        self.mock_stream.close.assert_called_once()
        self.assertNotIn('input', self.manager._active_streams)
        self.assertIsNone(self.manager.input_stream)

    def test_close_all_streams(self):
        """Test closing all active streams"""
        # Create both input and output streams
        self.manager.initialize_streams()
        
        # Close all streams
        self.manager.close_all_streams()
        
        # Verify all streams were properly closed
        self.assertEqual(self.mock_stream.stop_stream.call_count, 2)
        self.assertEqual(self.mock_stream.close.call_count, 2)
        self.assertEqual(len(self.manager._active_streams), 0)
        self.assertIsNone(self.manager.input_stream)
        self.assertIsNone(self.manager.output_stream)

    def test_terminate(self):
        """Test termination of PyAudio instance"""
        # Create some streams first
        self.manager.initialize_streams()
        
        # Terminate
        self.manager.terminate()
        
        # Verify everything was properly cleaned up
        self.mock_stream.stop_stream.assert_called()
        self.mock_stream.close.assert_called()
        self.mock_pyaudio.terminate.assert_called_once()
        self.assertEqual(len(self.manager._active_streams), 0)

    def test_get_default_device_info(self):
        """Test getting default device information"""
        input_info = self.manager.get_default_input_device_info()
        output_info = self.manager.get_default_output_device_info()
        
        self.assertEqual(input_info, self.mock_device_info)
        self.assertEqual(output_info, self.mock_device_info)

    def test_get_sample_size(self):
        """Test getting sample size for a format"""
        size = self.manager.get_sample_size(mock_pyaudio.paInt16)
        
        self.mock_pyaudio.get_sample_size.assert_called_once_with(mock_pyaudio.paInt16)
        self.assertEqual(size, 2)

if __name__ == '__main__':
    unittest.main()
