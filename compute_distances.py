import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import random
import pandas as pd
from multiprocessing import get_context
import traceback


def parallel_distance(input, poi):
    videoname, row = input
    context = row['context']
    ## Double check the poi is correct
    assert row['poi'] == poi

    ## Default value is NaN
    value = np.nan

    ## load feature file if exists
    features_file = get_features_filename(videoname)
    if os.path.isfile(features_file):
        try:
            features = np.load(features_file)
            ## Get all the features with different contexts from the test audio
            references = np.concatenate([list_dict[k] for k in list_dict if k != context], 0)

            if opt.strategy == 'ms':
                value = np.min(np.sum(np.square(features[:1, :] - references), -1), -1)
            elif opt.strategy == 'cb':
                references = np.mean(references, 0)
                value = np.sum(np.square(features[0, :] - references), -1)
        except:
            traceback.print_exc()
    else:
        print(f'feature file {features_file} does not exist!')
    return videoname, value


def split_audio_to_new(input_file, output_folder="hadar_ref", segment_duration=6000):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Calculate the total duration of the audio in milliseconds
    total_duration = len(audio)

    # Calculate the number of segments needed
    num_segments = total_duration // segment_duration

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Split the audio into segments
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration
        segment = audio[start_time:end_time]

        # Generate the output file name
        output_file = os.path.join(output_folder, f'{os.path.splitext(os.path.basename(input_file))[0]}_seg_{i + 1}.wav')
        # Export the segment as a new audio file
        segment.export(output_file, format='wav')

    if num_segments==0:
        output_file = os.path.join(output_folder,
                                   f'{os.path.splitext(os.path.basename(input_file))[0]}.wav')
        # Export the segment as a new audio file
        segment.export(output_file, format='wav')


    print(f'{num_segments} segments created successfully.')


def split_all_ref(input_folder, output_folder):
    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):  # You can adjust the file extension if needed
            input_file_path = os.path.join(input_folder, filename)
            # output_file_path= os.path.join(output_folder, os.path.splitext(filename)[0])

            # Split ref audio file and save in a subfolder- if too long
            if filename.startswith("ref_"):
                split_audio_to_new(input_file=input_file_path, output_folder=output_folder)
            else:
                output_file = os.path.join(output_folder,
                                           f'{os.path.splitext(os.path.basename(input_file_path))[0]}.wav')
                audio = AudioSegment.from_file(input_file_path)
                audio.export(output_file, format='wav')
                
def create_metadata_csv(input_folder, output_csv, poi="Hadar", context="0"):  # will run on the splitted audio's dir
    import os
    import csv
    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: The input folder '{input_folder}' does not exist.")
        return

    # Create a CSV file for metadata
    with open(output_csv, mode='w', newline='') as csv_file:
        fieldnames = ['videoname', 'filepath', 'poi', 'context', 'label', 'in_tst', 'in_ref']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write the header row to the CSV file
        writer.writeheader()

        # Iterate over all files in the input folder
        for filename in os.listdir(input_folder):
            if filename.endswith(".wav"):  # You can adjust the file extension if needed
                videoname = os.path.splitext(filename)[0]
                filepath = os.path.abspath(os.path.join(input_folder, filename))
                label = 0 # Default label is 0- used for evaluation only!
                in_tst = 1 if filename.startswith('tst_') else 0
                in_ref = 1 if filename.startswith('ref_') else 0

                # Write the row to the CSV file
                writer.writerow({'videoname': videoname, 'filepath': filepath, 'poi': poi,
                                 'context': context, 'label': label, 'in_tst': in_tst, 'in_ref': in_ref})


if __name__ == '__main__':
    # input_folder = "datasetSpeechByUs"
    # output_folder = "HadarDataset"
    # output_csv = 'CSVs/dataset.csv'
    # poi = "Hadar"
    # context = "0"
    # split_all_ref(input_folder, output_folder)
    # create_metadata_csv(output_folder,output_csv,poi,context)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-csv', type=str, default=None)
    parser.add_argument('--dataset-name', type=str, default=None)
    parser.add_argument("--features-folder", type=str, default="./features")
    parser.add_argument("--scores-folder", type=str, default="./scores")
    parser.add_argument("--strategy", type=str, default="ms")
    parser.add_argument("--num-threads", type=int, default=16)
    opt = parser.parse_args()
    assert opt.strategy in ['cb', 'ms'], "Strategy not supported, please select one between Centroid-Based (cb) or " \
                                         "Multi-Similarity (ms)."
    def get_features_filename(filename):
        return os.path.join(opt.features_folder, opt.dataset_name, f"{filename}.npy")

    ## Load the csv and split it into two csv, test and reference
    df = pd.read_csv(opt.dataset_csv, index_col='videoname')
    df.index = df.index.astype(str)
    df_ref = df[df['in_ref'] != 0][['poi', 'context', 'label']]
    df_tst = df[df['in_tst'] != 0][['poi', 'context']]
    df_tst['value'] = np.nan
    assert np.all(df_ref['label'] == 0) #0 is real
    del df

    ## A list of all the unique identities
    list_poi = df_ref['poi'].unique()
    ## for each identity
    for poi in list_poi:
        # we get the test and reference list of files for the given poi
        df_ref_poi = df_ref[df_ref['poi'] == poi]
        df_tst_poi = df_tst[df_tst['poi'] == poi]
        ## If one of the two lists are empty, we skip this poi (no file to tests or no reference available)
        if (len(df_ref_poi) == 0) or (len(df_tst_poi) == 0):
            continue

        ## We split features given their context. When testing, we only take files from a different context to avoid polatization
        list_dict = {k: list() for k in df_ref_poi['context'].unique()}

        for videoname, row in tqdm(df_ref_poi.iterrows(), total=len(df_ref_poi), desc='Loading ' + poi):
            features_file = get_features_filename(videoname)
            context = row['context']
            ## Check the POI is correct
            assert row['poi'] == poi

            try:
                list_dict[context].append(np.load(features_file))
            except:
                print('ERROR with ref file: ', poi, context, features_file)

        ## Concatenate the list into an array
        for k in list_dict:
            list_dict[k] = np.concatenate(list_dict[k], 0)
        print('DONE Loading ' + poi, flush=True)

        # with ctx.Pool(opt.num_threads) as pool:
        #     out = pool.imap_unordered(parallel_distance, df_tst_poi.iterrows())
        #     for video, value in tqdm(out, total=len(df_tst_poi), desc='Testing ' + poi):
        #         df_tst.loc[video, 'value'] = value

        for video, row in tqdm(df_tst_poi.iterrows(), total=len(df_tst_poi), desc='Testing ' + poi):
            # video = row['video']  # Assuming 'video' is the column name
            value = parallel_distance((video,row), poi)  # Assuming parallel_distance accepts a row as an argument
            df_tst.loc[video, 'value'] = value[1]

        print('DONE Testing ' + poi, flush=True)

    ## Save result to file
    df_tst.to_csv(os.path.join(opt.scores_folder, f"{opt.dataset_name}_{opt.strategy}.csv"))
