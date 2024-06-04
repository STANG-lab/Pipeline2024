#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os.path
import csv
import subprocess
from abstract_featurizer import Featurizer


# Update audio_path and output_path to desired preferences
# 
# Config file....TBD

# In[18]:


# smile_path = '/Volumes/Alex_R_Music_ssD/Research/ac_pipe/opensmile-3.0.1-macos-x64/bin/SMILExtract'
# #config_path = '/Volumes/Alex_R_Music_ssD/Research/ac_pipe/opensmile-3.0.1-macos-x64/config/is09-13/IS13_ComParE.conf'
# config_path = '/Volumes/Alex_R_Music_ssD/Research/ac_pipe/opensmile-3.0.1-macos-x64/config/myIS13_ComParE_8K.conf'
# #config_path = '/Volumes/Alex_R_Music_ssD/Research/ac_pipe/opensmile-3.0.1-macos-x64/config/demo/demo1_energy.conf'
#
# output_path = "/Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/smile_out/remora_test/openSMILE_output"
# log_path = "/Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/smile_out/remora_test/log_files"
# #output_path="/Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/smile_out/remora_std_spellchecked_batch_1+2/"
# # Path(output_path).mkdir(parents=True, exist_ok=True)
# # Path(log_path).mkdir(parents=True, exist_ok=True)
#
# audio_df_path = "/Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/snr_out/rem_batch1_snr_final.csv" # This should stay here
#
# audio_path = '/Volumes/Alex_R_Music_ssD/Research/ac_pipe/data/remora_test/audio_by_task'
# denoised_path = "/Volumes/Alex_R_Music_ssD/Research/ac_pipe/data/remora_test/audio_by_task/denoised"
#
# audio_files = [f for f in os.listdir(audio_path) if os.path.isfile(os.path.join(audio_path, f)) and not f.startswith('.')] # infs hidden files and ignores folders
# denoised_files = [f for f in os.listdir(denoised_path) if os.path.isfile(os.path.join(denoised_path, f)) and not f.startswith('.')] # infs hidden files and ignores folders
# audio_df = pd.read_csv(audio_df_path)



# STILL NEED TO FINISH CONFIGURING BELOW (WIP)

# Make List of Files to run through openSMILE

# In[19]:


# mask_denoised = audio_df['denoised'] == True
#
# non_denoised_df = audio_df[~mask_denoised]  # Files that were not denoised
# denoised_df = audio_df[mask_denoised]  # Files that were denoised
#
# mask_eligible = denoised_df['eligible'] == True
# denoised_eligible_df = denoised_df[mask_eligible]
#
# non_denoised_files = non_denoised_df['file_name'].tolist()
# denoised_files = denoised_eligible_df['file_name'].tolist()
# all_files = non_denoised_files + denoised_files
#
# print("# of Eligible Non-Denoised Files:")
# print(len(non_denoised_files))
# print("# of Eligible Denoised Files:")
# print(len(denoised_files))
#
# print("Eligible Non-Denoised Files")
# print(non_denoised_files)
# print("\nEligible Denoised Files:")
# print(denoised_files)
#
# # Sanity check - should be no overlapping files
# print("\nOverlapping files between lists (should be none):")
# print(len(list(set(non_denoised_files).intersection(denoised_files))))


# Run Audio Files through openSMILE
# ./bin/SMILExtract -C config/demo/demo1_energy.conf -I /Volumes/Alex_R_Music_ssD/Research/ac_pipe/data/remora_test/audio_by_task/remora_10308_T1_10308T1ER40_v2023_ER40_ER40.wav -O /Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/smile_out/test.csv

# In[21]:

class AcousticFeaturizer(Featurizer):
    def __init__(self, outdirs, aud_target, smile_path, config_path, smile_log_out=True):
        super().__init__(self, outdirs)
        self.audio_path = aud_target
        # Alex's original code distinguishes two sources, denoised and raw files.
        # Instead, we analyze whichever is our audio target.

        # SMILE SPECIFIC CONFIG
        self.smile_command = smile_path
        self.config_path = config_path

        #Processing
        self.smile_on_files()
        self.process_smiles()



    def smile_on_files(self):
        # Calls acoustic extractor.
        sub_results = []

        for file in self.audio_path.glob("*"):
            file_path = os.path.join(audio_path, file)
            outfile_path = os.path.join(output_path, file.split('.wav')[0]+'.csv')

            args = [self.smile_command, "-C", self.config_path, "-I", file_path, "-O", outfile_path]
            print("Processing file: ", file_path)
            sub_results.append(subprocess.run(args, capture_output=True))

        return sub_results

    def process_smiles(self):
        # pre_smiles = [f for f in os.listdir(smile_files_path) if
        #               os.path.isfile(os.path.join(smile_files_path, f)) and not f.startswith(
        #                   '.')]  # infs hidden files and ignores folders

        pre_smiles = self.smile_out.glob("*.csv")
        # print("Total # openSMILE files:")
        # print(len(pre_smiles))

        for smile_name in pre_smiles:
            print("Processing ", smile_name)

            post_smile = list()

            # DEBUG - TODO remove later
            # if smile_name != 'remora_10308_T1_10308T1ER40_v2023_ER40_ER40.csv':
            # continue

            # Read corresponding Lab file into list
            smile_path = self.smile_out / smile_name.name
            # smile_path = os.path.join(smile_files_path, smile_name)
            lab_name = smile_name.name.split('.')[0] + "_LABELED.lab"
            # lab_path = os.path.join(lab_files_path, lab_name)
            lab_path = self.sad_postlab / lab_name

            with open(smile_path, newline='') as f:
                reader = csv.reader(f, delimiter=";")
                smile = list(reader)

            with open(lab_path, newline='') as f:
                lab = csv.reader(f, delimiter="\t")
                # lab = list(reader)

            # Create header for labeled smile file
            header = list()
            for label in smile[0]:
                # Insert speaker and speech-type after frameTime
                header.append(label)
                if label == 'frameTime':
                    header.append('speaker')
                    header.append('type')

            post_smile.append(header)

            i = 1  # Skip header

            # Iterate through lab file lines and label corresponding smile frames
            for line in lab:
                start = float(line[0])
                end = float(line[1])
                type = line[3]
                speaker = line[4]

                for row in smile[i:]:
                    new_row = list()
                    frame_index = int(row[0])
                    frame_time = float(row[1])

                    if frame_time >= end:
                        continue

                    new_row.append(frame_index)
                    new_row.append(frame_time)
                    new_row.append(speaker)
                    new_row.append(type)
                    for feature in row[2:]:
                        new_row.append(feature)
                    post_smile.append(new_row)
                    i += 1

            # Save post_smile file
            lab_smile_name = smile_name.name.split(".")[0] + '_LABELED.csv'
            # path = os.path.join(output_path, smile_name + '_LABELED.csv')
            with open(self.smile_labeled_out / lab_smile_name, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerows(post_smile)
        print("DONE!")


# def log_results_info(sub_results, log_name):
#     log_name = os.path.join(log_path,  'log.txt')
#     with open(log_name, 'w', newline='') as f:
#         for result in sub_results:
#             print("\nCommand:")
#             print(result.args)
#             print("Stdout:")
#             print(result.stdout)
#             print("Stderr:")
#             print(result.stderr)
#             f.write("\nCommand: ")
#             f.write(f"{result.args}\n")
#             f.write("Stdout:\n")
#             f.write(f"{result.stdout}\n")
#             f.write("Stderr:\n")
#             f.write(f"{result.stderr}\n")
#
#         print("\n\nExpected # Files CSVs Generated:")
#         f.write("\n\nExpected # Files CSVs Generated: ")
#         print(len(all_files))
#         f.write(f"{len(all_files)}\n")
#         print("# of Files Created")
#         f.write("# of Files Created: ")
#         all_output_files = [f for f in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, f)) and not f.startswith('.')] # infs hidden files and ignores folders
#         print(len(all_output_files))
#         f.write(f"{len(all_output_files)}")


# Label openSMILE output using lab files
# - can do speech only vs speech and non-speech and compare results

# In[5]:


# smile_files_path = "/Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/smile_out/remora_test/openSMILE_output"
# lab_files_path = "/Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/sad_out/test/post_labs"
# output_path = "/Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/smile_out/remora_test/openSMILE_output_labeled"
# Path(output_path).mkdir(parents=True, exist_ok=True)
#
# pre_smiles = [f for f in os.listdir(smile_files_path) if os.path.isfile(os.path.join(smile_files_path, f)) and not f.startswith('.')] # infs hidden files and ignores folders
#
# print("Total # openSMILE files:")
# print(len(pre_smiles))


# In[14]:

        # # smile_path = '/Volumes/Alex_R_Music_ssD/Research/ac_pipe/opensmile-3.0.1-macos-x64/bin/SMILExtract'
        # # config_path = '/Volumes/Alex_R_Music_ssD/Research/ac_pipe/opensmile-3.0.1-macos-x64/config/is09-13/IS13_ComParE.conf'
        # # config_path = '/Volumes/Alex_R_Music_ssD/Research/ac_pipe/opensmile-3.0.1-macos-x64/config/myIS13_ComParE_8K.conf'
        # # config_path = '/Volumes/Alex_R_Music_ssD/Research/ac_pipe/opensmile-3.0.1-macos-x64/config/demo/demo1_energy.conf'
        #
        #
        #
        # self.smile_logs = outdirs['openSMILE/logs']
        # # log_path = "/Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/smile_out/remora_test/log_files"
        #
        #
        # # output_path="/Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/smile_out/remora_std_spellchecked_batch_1+2/"
        # # Path(output_path).mkdir(parents=True, exist_ok=True)
        # # Path(log_path).mkdir(parents=True, exist_ok=True)
        #
        # self.audio_df_path = outdirs["Features"] / "snr_final.csv"
        # # audio_df_path = "/Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/snr_out/rem_batch1_snr_final.csv"
        #
        # self.audio_path = aud_target
        # # Alex's original code distinguishes two sources, denoised and raw files.
        # # Instead, we analyze whichever is our audio target.
        #
        # # audio_path = '/Volumes/Alex_R_Music_ssD/Research/ac_pipe/data/remora_test/audio_by_task'
        #
        # # denoised_path = outdirs["Raw/denoised"]
        # # denoised_path = "/Volumes/Alex_R_Music_ssD/Research/ac_pipe/data/remora_test/audio_by_task/denoised"
        #
        # # self.lab_files_path = outdirs["SAD/postlab"]
        # # lab_files_path = "/Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/sad_out/test/post_labs"
        #
        # # The folders below are populated by this object, as opposed to the previous which are pre-supposed.
        # self.smile_output_path = outdirs["openSMILE/Output"]
        # # output_path = "/Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/smile_out/remora_test/openSMILE_output"
        # self.smile_labeled_path = outdirs["openSMILE/LabOutput"]
        # # output_path = "/Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/smile_out/remora_test/openSMILE_output_labeled"



# Generate Descriptive Statistics for each openSMILE CSV and save in metadata CSV file

# Visualize Some Features using GnuPlot (https://audeering.github.io/opensmile/get-started.html#extracting-features-with-opencv)?
