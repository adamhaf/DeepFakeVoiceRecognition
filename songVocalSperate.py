import os
import subprocess


def separate_audio(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".mp3"):
            input_file = os.path.join(input_dir, filename)
            output_subdir = os.path.join(output_dir, os.path.splitext(filename)[0])
            os.makedirs(output_subdir, exist_ok=True)

            subprocess.run(["spleeter", "separate", "-o", output_subdir, input_file])


if __name__ == "__main__":
    input_dir = "C:\\Users\\adamm\\finalProject\\singerDataSet\\OmerAdam\\afterAllTheYears"
    output_dir = "C:\\Users\\adamm\\finalProject\\singerDataSet\\OmerAdam\\afterAllTheYears\\vocal"
    separate_audio(input_dir, output_dir)
