import wave
import numpy as np

from pydub import AudioSegment


def cut_mp3(input_file, output_file, start_time, end_time):
    # Load the MP3 file
    audio = AudioSegment.from_mp3(input_file)

    # Set start and end times in milliseconds
    start_ms = start_time * 1000
    end_ms = end_time * 1000

    # Trim the audio
    trimmed_audio = audio[start_ms:end_ms]

    # Export the trimmed audio to a new file
    trimmed_audio.export(output_file, format="mp3")


def cut_wav(input_file, output_file, start_time, end_time):
    # Open the input WAV file
    with wave.open(input_file, 'rb') as wave_file:
        # Get the parameters of the input WAV file
        params = wave_file.getparams()
        num_channels = params.nchannels
        sample_width = params.sampwidth
        frame_rate = params.framerate
        num_frames = params.nframes

        # Calculate start and end frames
        start_frame = int(start_time * frame_rate)
        end_frame = int(end_time * frame_rate)

        # Ensure end_frame is within the bounds of the audio
        end_frame = min(end_frame, num_frames)

        # Set up the output WAV file
        with wave.open(output_file, 'wb') as output_wave:
            output_wave.setparams(params)

            # Read frames from input file and write to output file
            wave_file.setpos(start_frame)
            frames_to_copy = end_frame - start_frame
            frames = wave_file.readframes(frames_to_copy)
            output_wave.writeframes(frames)



if __name__ == "__main__":
    input_file = "processing\eyalAndOmerSmackThat.mp3"
    output_file = "OmerSmackThat.mp3"
    start_time = 33  # Start time in seconds
    end_time = 43  # End time in seconds

    cut_mp3(input_file,output_file,start_time,end_time)
