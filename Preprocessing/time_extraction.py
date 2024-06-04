#!/usr/bin/env python
# coding: utf-8

# This file will extract time-based features, such as speech latency, speaking rate, # short/med/long pauses, pause duration, median/mean speach and pause duration
# 
# SETUP:
#     Need to download HTK, and move the tarball into ldc-bpcsad/src,
#     then build a docker image
#     Then need to run the docker container: e.g. docker run --rm -v /Volumes/Alex_R_Music_ssD/Research/ac_pipe:/ac_pipe ldc-bpcsad "ldc-bpcsad --output-dir /ac_pipe/sad_output /ac_pipe/data/10311_BL_2_1.wav /ac_pipe/data/10455_BL_2_1.wav"
#         - This command runs the ldc-bpcsad tool
#         --rm: cleans up the container and removes the file system when the container exits
#         -v: provides container access to the specified volume ac_pipe that is visible within the container as /ac_pipe
#         ldc-bpcsad is the image the container is derived from
#        "ldc-bpcsad --output-dir /ac_pipe/output /ac_pipe/data/10311_BL_2_1.wav" is the command to run (runs ldc-bpcsad on listed audio files and outputs to specified directory) 
# 
#     - FOR EASIER FUNCTIONALITY....used venv with python instead of docker
#         - used the .venv (/Volumes/Alex_R_Music_ssD/Research/ac_pipe/.venv) used for the pre_processing and such with python 3.9
#         - HTK was installed into the .venv/bin/
#         - directory path exported to PATH
#         - so, do "source .venv/bin/activate" to active the virtual environment
#     
# Output:
#     New lab files will be created and labeled, so output will be tab-separate values in format: StartTime Endtime Speech/Non-Speech Type Speaker
#         - Type: Speech for speech. For non-speech, pause vs latency (changing speakers)
#             - NOTES: "NA_SPEAKER" indicates prior speaker is incorrectly labeled and therefore latency vs pause cannot be determined
#                      "NA_SKIP" indicates prior speaker is skipped due to lab line not having corresponding megasegment
#                      "Interviewer+Patient" indicates overlapping segment during which the interviewer was originally speaking and the patient interrupted
#         - Speaker: will contain the speaker as per the transcript file verbatim, including if the speaker was mislabled or left blank
#             - "SKIP" is used when lab file does not have corresponding megasegment
#         - "NA" also used to denote non-speech that begins or completes lab files as type of non-speech cannot be determined
# 
# 
# 
# Log Output:
#     Megasegment folder will contain the megasegments derived from each transcript file to assist with possible debugging
# 
#     "log.tsv" will contain all errors/warnings (i.e. overlapping speakers, mislabeled speakers, lab lines SKIPPED)

# In[1]:


import pandas as pd
import subprocess
# from pathlib import Path
import glob
import csv
from abstract_featurizer import Featurizer

# SET PATHs accordingly

# In[2]:
# Takes in multi-dimensional array, labs, where lab[0] is file name, and creates list of unique filenames
def unique_names(labs):
    # Create an empty set to store unique filenames
    unique_filenames = set()

    # Iterate over the list and extract unique filenames
    for lab in labs:
        unique_filenames.add(lab[0])

    # Convert the set of unique filenames back to a list
    return list(unique_filenames)


# Determines if speaker is known label
def is_unknown_speaker(speaker):
    speaker = speaker.lower().strip()
    valid_speakers = {"interviewer", "subject", "interviewer+subject", "subject+interviewer", "other", "SPEAKER_00",
                      "SPEAKER_01"}
    if speaker in valid_speakers:
        return False
    else:
        return True


# def set_paths(audio_path, text_path, outdirs):
#     return audio_path, text_path, outdirs["SAD/prelab"], outdirs["SAD/postlab"], outdirs["SAD/megasegs"], outdirs["SAD/logs"], outdirs["Features"]

# path_to_audio = '/Volumes/Alex_R_Music_ssD/Research/ac_pipe/data/remora_by_task/remora_audio_task'
# path_to_trans = '/Volumes/Alex_R_Music_ssD/Research/ac_pipe/data/remora_by_task/1b_std_transcripts_spellchecked_1+2'


# Make export directories if non-existent
# pre_output_path= "/Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/sad_out/remora_std_spellchecked_batch_1+2/pre_labs"
# output_path = "/Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/sad_out/remora_std_spellchecked_batch_1+2/post_labs"
# mega_path = "/Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/sad_out/remora_std_spellchecked_batch_1+2/log_files/megasegments"
# log_path = "/Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/sad_out/remora_std_spellchecked_batch_1+2/log_files"
#
# features_path = "/Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/sad_out/remora_std_spellchecked_batch_1+2/features"

class TimeFeaturizer(Featurizer):
    def __init__(self, outdirs, audio_path, text_path):
        super().__init__(outdirs)
        self.audio_path = audio_path
        self.text_path = text_path
        self.featurize()
    def featurize(self):
        # Initial SAD run on raw interviewer + participant WAV files
        self.run_SAD()
        self.get_time_features()

    def run_SAD(self):
        # SAD run on WAV files
        # TODO: MAKE THIS WORK WITH MORE FILETYPES

        list_of_audio = glob.glob(self.audio_path + '/*.wav')
        # list_of_trans = glob.glob(self.text_path + '/*.txt')

        # print(len(list_of_audio), " audio files.")
        # print(len(list_of_trans), " transcript files.")

        args = ["ldc-bpcsad", "--output-dir", self.sad_prelab, "--nonspeech", "0.250", "--speech", "0.250"]
        args = args + list_of_audio
        subprocess.run(args)
        print("DONE!")

    def label_lab_file(self, list_lab, list_mega, offset, file):

        # Create columns for task, speaker, pause/latency/speech
        lab_iter = 0  # lab iter
        mega_iter = 0  # mega iter

        new_list_lab = list()

        unlabeled_labs = list()

        # Label all speech
        for mega in list_mega:
            start = float(mega[0]) - offset
            end = float(mega[1]) - offset
            speaker = mega[2]

            for lab in list_lab[lab_iter:]:
                # Normalize all time values
                lab[0] = float(lab[0])
                lab[1] = float(lab[1])
                lab[2] = str(lab[2])

                # Process speech
                if lab[2] == 'speech':
                    # Completed megasegment
                    if lab[0] > end:
                        break

                    # CONTAINED in megasegment
                    if lab[0] >= start and lab[1] <= end:
                        new_list_lab.append([lab[0], lab[1], lab[2], 'speech', speaker])

                    # PRECEDES megasegment
                    elif lab[0] < start and (lab[1] <= end and lab[1] > start):
                        new_list_lab.append([lab[0], lab[1], lab[2], 'speech', speaker])

                    # EXCEEDS or PRECEDES AND EXCEDES megasegment
                    elif lab[1] > end:
                        # Determine extent of overlap
                        overlap = 0
                        for k in range(1, len(list_mega) - mega_iter):
                            if lab[1] >= list_mega[mega_iter + k][0] - offset:
                                overlap = k
                            else:
                                break

                        # Only involves single megasegment (lab endtime < next mega start time)
                        if overlap == 0:
                            new_list_lab.append([lab[0], lab[1], lab[2], 'speech', speaker])
                        # Overlaps with subsequent megasegment(s)
                        else:
                            # Divide lab line into multiple lab lines based on megasegment
                            new_list_lab.append(
                                [lab[0], list_mega[mega_iter][1] - offset, lab[2], 'speech', list_mega[mega_iter][2]])
                            for k in range(1, overlap):
                                new_list_lab.append(
                                    [list_mega[mega_iter + k][0] - offset, list_mega[mega_iter + k][1] - offset, lab[2],
                                     'speech', list_mega[mega_iter + k][2]])
                            new_list_lab.append([list_mega[mega_iter + overlap][0] - offset, lab[1], lab[2], 'speech',
                                                 list_mega[mega_iter + overlap][2]])
                    else:  # Lab does not overlap with megasegment
                        unlabeled_labs.append([file, str(lab[0]), str(lab[1])])
                        new_list_lab.append([lab[0], lab[1], lab[2], 'speech', 'SKIP'])

                else:
                    new_list_lab.append([float(lab[0]), float(lab[1]), str(lab[2])])

                lab_iter += 1

            mega_iter += 1

        # Process non-speech
        # Edge case for starting on non-speech (can occur if initial speech < 0.5s)
        if new_list_lab[0][2] == 'non-speech':
            last_speaker = new_list_lab[1][4]
            skip_first = True
            new_list_lab[0].append('NA')
            new_list_lab[0].append('NA')
        else:
            last_speaker = new_list_lab[0][4]
            skip_first = False
        i = 0
        end = len(new_list_lab)
        for lab in new_list_lab:
            i += 1
            if skip_first:
                skip_first = False
                continue
            if i == end:
                # Edge for ending on non-speech
                if lab[2] == 'non-speech':
                    lab.append('NA')
                    lab.append('NA')
                break

            if lab[2] == 'non-speech':
                # Pauses
                if new_list_lab[i][4] == last_speaker:
                    lab.append('pause')
                    lab.append(last_speaker)
                # Latency (i.e. new speaker)
                else:
                    if last_speaker == "SKIP":
                        lab.append('NA_SKIP')
                    elif is_unknown_speaker(last_speaker):
                        lab.append('NA_SPEAKER')
                    else:
                        lab.append('latency')
                    last_speaker = new_list_lab[i][4]
                    lab.append(last_speaker)
            else:
                last_speaker = lab[4]

        return new_list_lab, unlabeled_labs
    def log_problems(self, list_of_error, list_of_overlaps, list_of_extra_speakers, unlabeled_labs):
        print("\n\nTranscript files not found:\n")
        for file in list_of_error:
            print(file)
        print("\n\nFiles containing OVERLAPPING speakers:\n")
        print("Transcript\tLine #\n")
        for line in list_of_overlaps:
            print(line[0], "\t", str(line[1]))
        print("\n\nFiles containing UNEXPECTED speakers:\n")
        print("Transcript\tSpeaker\tLine #\n")
        for line in list_of_extra_speakers:
            print(line[0], "\t", line[1], "\t", line[2])
        unlabeled_files = unique_names(unlabeled_labs)
        print("\n\nFiles containing UNLABELED Lab Lines:\n")
        for file in unlabeled_files:
            print(file)
        print("DONE!")

        log_name = self.sad_logs + '/' + 'log.tsv'
        with open(log_name, 'w', newline='') as f:
            f.write("Files Not Found:\n")
            for line in list_of_error:
                f.write(f"{line}\n")

            writer = csv.writer(f, delimiter='\t')

            f.write("\n\nFiles with OVERLAPPING Found:\n")
            f.write("Transcript\tLine #\n")
            writer.writerows(list_of_overlaps)

            f.write("\n\nFiles with UNEXPECTED speakers Found:\n")
            f.write("Transcript\tSpeaker\tLine #\n")
            writer.writerows(list_of_extra_speakers)


            f.write("\n\nFiles with UNLABELED labs (no overlap megasegments):\n")
            f.write("File, Lab Start, Lab End\n")
            writer.writerows(unlabeled_labs)

    def get_time_features(self):
        '''
        path_to_audio = '/Volumes/Alex_R_Music_ssD/Research/ac_pipe/data/remora_test/audio_by_task'
        path_to_trans = '/Volumes/Alex_R_Music_ssD/Research/ac_pipe/data/remora_test/transcripts_by_task'


        # Make export directories if non-existant
        pre_output_path= "/Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/sad_out/test/pre_labs"
        output_path= "/Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/sad_out/test/post_labs"
        mega_path = "/Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/sad_out/test/log_files/megasegments"
        log_path = "/Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/sad_out/test/log_files"

        Path(output_path).mkdir(parents=True, exist_ok=True)
        Path(pre_output_path).mkdir(parents=True, exist_ok=True)
        Path(log_path).mkdir(parents=True, exist_ok=True)
        Path(mega_path).mkdir(parents=True, exist_ok=True)
        '''
        # list_of_lab_files = glob.glob(pre_output_path  '/*.lab')
        list_of_lab_files = list(self.sad_prelab.glob("*.lab"))
        list_of_trans_files = list(self.text_path.glob("*.txt"))
        # list_of_trans_files = glob.glob(path_to_trans + '/*.txt')
        list_of_error = list()
        list_of_extra_speakers = list()
        list_of_overlaps = list()
        unlabeled_labs = list()

        # print(len(list_of_lab_files), " pre-processed lab files.")

        for lab_path in list_of_lab_files:
            lab_name = lab_path.split('/')[-1].split('.lab')[0]
            trans_path = self.text_path + '/' + lab_name + '.txt'
            print("Processing ", lab_path)

            # check if the trans file in the list of transfiles with old names
            if trans_path in list_of_trans_files:
                # Use list for efficiency instead of pandas dataframe
                with open(trans_path, newline='') as f:
                    reader = csv.reader(f, delimiter="\t")
                    list_tran = list(reader)
                with open(lab_path, newline='') as f:
                    reader = csv.reader(f, delimiter="\t")
                    list_lab = list(reader)


                # Gather all megasegments
                list_mega = list()
                start_time = float(list_tran[0][2])
                offset = start_time  # Used to account for transcript timestamps not starting at 0
                end_time = float(list_tran[0][3])
                last_speaker = list_tran[0][1]
                list_len = len(list_tran)
                i = 0
                for row in list_tran:
                    i += 1
                    #Remove white space and assign timestamps as float
                    row[2] = float(row[2])
                    row[3] = float(row[3])
                    row[0] = str(row[0]).strip()
                    row[1] = str(row[1]).strip()

                    # Log any files with speakers other than "Interviewer" or "Subject" or "Patient"
                    if is_unknown_speaker(row[1].lower()):
                        print("Non-interviewer/subject label '", row[1], "' on line ", i, " found in file: ", trans_path.split('/')[-1])
                        list_of_extra_speakers.append([trans_path.split('/')[-1], str(row[1]), str(i)])


                    # If same speaker, update new end timestamp
                    if row[1] == last_speaker:
                        end_time = row[3]


                    # If different speaker, process lab file correlating to current megasegment, determine if OVERLAP present
                    else:
                        # NO OVERLAP
                        if row[2] > end_time:
                            # Save current "megasegment"
                            mega_entry = [float(start_time), float(end_time), str(last_speaker)]
                            list_mega.append(mega_entry)

                            # Update values for new megasegment
                            start_time = float(row[2])
                            end_time = float(row[3])
                            last_speaker = row[1]


                        else: #OVERLAP
                            # Log any files with overlapping speakers
                            print("OVERLAPPING speakers on line ", i, " found in file: ", trans_path.split('/')[-1])
                            list_of_overlaps.append([trans_path.split('/')[-1], i])


                            # If speaker 2's overlap is contained within speaker 1's speech create megasegment for Speaker 1
                            # and start a new megasegment with the overlap
                            if row[3] <=  end_time:
                                mega_entry = [float(start_time), float(row[2]), str(last_speaker)]
                                list_mega.append(mega_entry)

                                # Interrupting Contained megsegment
                                speaker = last_speaker + '+' + str(row[1])
                                mega_entry = [float(row[2]), float(row[3]), str(speaker)]
                                list_mega.append(mega_entry)

                                # New megasegment for original speaker (or next speaker if both speakers complete at same time)
                                if end_time == row[3] and i != list_len:
                                    last_speaker = list_tran[i][1]
                                start_time = float(row[3])

                            else:  # Speaker 2' overlap is NOT entirely contained within speaker 1's speech
                                mega_entry = [float(start_time), float(row[2]), str(last_speaker)]
                                list_mega.append(mega_entry)

                                # Overlapping megasegment
                                speaker = last_speaker + '+' + str(row[1])
                                mega_entry = [float(row[2]), end_time, str(speaker)]
                                list_mega.append(mega_entry)

                                # New megasegment for interruping speaker
                                start_time = end_time
                                last_speaker = str(row[1])
                                end_time = float(row[3])

                    # Label last row
                    if i >= list_len:
                        mega_entry = [float(start_time), float(end_time), str(last_speaker)]
                        list_mega.append(mega_entry)
                        break

                # Label lab file
                list_lab, unlabeled = self.label_lab_file(list_lab, list_mega, offset, trans_path.split('/')[-1])
                unlabeled_labs = unlabeled_labs + unlabeled

                # Output each transcript file's megasegment
                path = self.sad_megasegs / (lab_name + '.mega')
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerows(list_mega)

                # Output each newly labeled lab file
                path = self.sad_postlab / (lab_name + '_LABELED.lab')
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerows(list_lab)

            else:
                print('ERROR: ', trans_path, ' is not found')
                list_of_error.append(trans_path)

        self.log_problems(list_of_error, list_of_overlaps, list_of_extra_speakers, unlabeled_labs)

        all_features = ["filename", "study_name", "user_id", "timepoint", "sample_id", "version", "task", "stimulus",
                            "t_td_task", "t_td_speaker", "ti_td_speaker", "to_td_speaker", "tsk_td_speaker", "tb_td_speaker",
                            "t_td_speech", "ti_td_speech", "to_td_speech", "tsk_td_speech", "tb_td_speech",
                            "total_speech_time", "t_pd_speech", "ti_pd_speech", "to_pd_speech", "tsk_pd_speech", "tb_pd_speech",
                            "t_td_latency", "ti_td_latency", "to_td_latency", "tsk_td_latency", "tb_td_latency",
                            "t_pd_latency", "ti_pd_latency", "to_pd_latency", "tsk_pd_latency", "tb_pd_latency",
                            "t_td_pause", "ti_td_pause", "to_td_pause", "tsk_td_pause", "tb_td_pause",
                            "t_pd_pause", "ti_pd_pause", "to_pd_pause", "tsk_pd_pause", "tb_pd_pause",
                            "t_n_speech", "ti_n_speech", "to_n_speech", "tsk_n_speech", "tb_n_speech",
                            "t_n_latency", "ti_n_latency", "to_n_latency", "tsk_n_latency", "tb_n_latency",
                            "t_n_pause", "ti_n_pause", "to_n_pause", "tsk_n_pause", "tb_n_pause",
                            "t_n_pauseshort", "t_n_pausemed", "t_n_pauselong", "ti_n_pauseshort", "ti_n_pausemed", "ti_n_pauselong",
                            "t_md_speech", "t_mn_speech", "t_95_speech", "t_min_speech", "t_mx_speech",
                            "ti_md_speech", "ti_mn_speech", "ti_95_speech", "ti_min_speech", "ti_mx_speech",
                            "to_md_speech", "to_mn_speech", "to_95_speech", "to_min_speech", "to_mx_speech",
                            "tb_md_speech", "tb_mn_speech", "tb_95_speech", "tb_min_speech", "tb_mx_speech",
                            "t_md_pause", "t_mn_pause", "t_95_pause", "t_min_pause", "t_mx_pause",
                            "ti_md_pause", "ti_mn_pause", "ti_95_pause", "ti_min_pause", "ti_mx_pause",
                            "to_md_pause", "to_mn_pause", "to_95_pause", "to_min_pause", "to_mx_pause",
                            "tb_md_pause", "tb_mn_pause", "tb_95_pause", "tb_min_pause", "tb_mx_pause",
                            "t_md_latency", "t_mn_latency", "t_95_latency", "t_min_latency", "t_mx_latency",
                            "ti_md_latency", "ti_mn_latency", "ti_95_latency", "ti_min_latency", "ti_mx_latency",
                            "to_md_latency", "to_mn_latency", "to_95_latency", "to_min_latency", "to_mx_latency",
                            "tb_md_latency", "tb_mn_latency", "tb_95_latency", "tb_min_latency", "tb_mx_latency"]

        primary_features = ["filename", "study_name", "user_id", "timepoint", "sample_id", "version", "task", "stimulus",
                            "t_td_task", "t_td_speaker", "ti_td_speaker", "tb_td_speaker",
                            "t_td_speech", "ti_td_speech", "tb_td_speech",
                            "total_speech_time", "t_pd_speech", "ti_pd_speech", "tb_pd_speech",
                            "t_td_latency", "ti_td_latency", "tb_td_latency",
                            "t_pd_latency", "ti_pd_latency", "tb_pd_latency",
                            "t_td_pause", "ti_td_pause", "tb_td_pause",
                            "t_pd_pause", "ti_pd_pause", "tb_pd_pause",
                            "t_n_speech", "ti_n_speech", "tb_n_speech",
                            "t_n_latency", "ti_n_latency", "tb_n_latency",
                            "t_n_pause", "ti_n_pause", "tb_n_pause",
                            "t_n_pauseshort", "t_n_pausemed", "t_n_pauselong", "ti_n_pauseshort", "ti_n_pausemed", "ti_n_pauselong",
                            "t_md_speech", "t_min_speech", "t_mx_speech",
                            "ti_md_speech", "ti_min_speech", "ti_mx_speech",
                            "t_md_pause", "t_mx_pause",
                            "ti_md_pause", "ti_mx_pause",
                            "t_md_latency", "t_95_latency",
                            "ti_md_latency", "ti_95_latency"]

        supp_features = [strng for strng in all_features if strng not in primary_features]
        supp_features = ["filename", "study_name", "user_id", "timepoint", "sample_id", "version", "task", "stimulus"] + supp_features


        # In[18]:


        #Uncomment capture below and portion at end of script to write stdout/stderr to log files


        # In[ ]:



        features = list()

        header_txt = "filename, study_name, user_id, timepoint, sample_id, version, task, stimulus,\
                            t_td_task, t_td_speaker, ti_td_speaker, to_td_speaker, tsk_td_speaker, tb_td_speaker,\
                            t_td_speech, ti_td_speech, to_td_speech, tsk_td_speech, tb_td_speech,\
                            total_speech_time, t_pd_speech, ti_pd_speech, to_pd_speech, tsk_pd_speech, tb_pd_speech,\
                            t_td_latency, ti_td_latency, to_td_latency, tsk_td_latency, tb_td_latency,\
                            t_pd_latency, ti_pd_latency, to_pd_latency, tsk_pd_latency, tb_pd_latency,\
                            t_td_pause, ti_td_pause, to_td_pause, tsk_td_pause, tb_td_pause,\
                            t_pd_pause, ti_pd_pause, to_pd_pause, tsk_pd_pause, tb_pd_pause,\
                            t_n_speech, ti_n_speech, to_n_speech, tsk_n_speech, tb_n_speech,\
                            t_n_latency, ti_n_latency, to_n_latency, tsk_n_latency, tb_n_latency,\
                            t_n_pause, ti_n_pause, to_n_pause, tsk_n_pause, tb_n_pause,\
                            t_n_pauseshort, t_n_pausemed, t_n_pauselong, ti_n_pauseshort, ti_n_pausemed, ti_n_pauselong,\
                            t_md_speech, t_mn_speech, t_95_speech, t_min_speech, t_mx_speech,\
                            ti_md_speech, ti_mn_speech, ti_95_speech, ti_min_speech, ti_mx_speech,\
                            to_md_speech, to_mn_speech, to_95_speech, to_min_speech, to_mx_speech,\
                            tb_md_speech, tb_mn_speech, tb_95_speech, tb_min_speech, tb_mx_speech,\
                            t_md_pause, t_mn_pause, t_95_pause, t_min_pause, t_mx_pause,\
                            ti_md_pause, ti_mn_pause, ti_95_pause, ti_min_pause, ti_mx_pause,\
                            to_md_pause, to_mn_pause, to_95_pause, to_min_pause, to_mx_pause,\
                            tb_md_pause, tb_mn_pause, tb_95_pause, tb_min_pause, tb_mx_pause,\
                            t_md_latency, t_mn_latency, t_95_latency, t_min_latency, t_mx_latency,\
                            ti_md_latency, ti_mn_latency, ti_95_latency, ti_min_latency, ti_mx_latency,\
                            to_md_latency, to_mn_latency, to_95_latency, to_min_latency, to_mx_latency,\
                            tb_md_latency, tb_mn_latency, tb_95_latency, tb_min_latency, tb_mx_latency".split(',')
        headers = list()
        for head in header_txt:
            headers.append(head.strip())
        features.append(headers)

        # list_of_lab_files = glob.glob(post_lab_path + '/*.lab')
        for file in self.sad_postlab.glob("*.lab"):
            filename = file.name.split('/')[-1].split('.lab')[0]
            print(file, ' is being processed...')

            #DEBUG-REMOVE LATER TODO
            #if filename != 'remora_13101_T2_13101T201_v2023_BTW_none_LABELED':
                #continue

            file_split = filename.split('_')

            labs_df = pd.read_csv(file,
                                   header=None,
                                   sep='\t',
                                   names=['start', 'end', 'audio', 'type', 'speaker'])

            with open(file, newline='') as f:
                reader = csv.reader(f, delimiter="\t")
                labs_list = list(reader)

            study_name = file_split[0]
            user_id = file_split[1]
            timepoint = file_split[2]
            sample_id = file_split[3]
            version = file_split[4]
            task = file_split[5]
            stimulus = file_split[6]
            labs_df['duration'] = labs_df['end'] - labs_df['start']

            # Group by speech-type and speaker, and aggregate the durations
            durations = labs_df.groupby(['speaker', 'type'])['duration'].sum().reset_index()

            # Group by speech-type and speaker, and count occurrences
            counts = labs_df.groupby(['speaker', 'type']).size().reset_index(name='count')


            # Speaking Time Features
            try:
                t_td_speech = durations[(durations['type'] == 'speech') & (durations['speaker'] == 'Subject')]['duration'].values[0]
            except IndexError:
                t_td_speech = 0
                print("WARNING: ", file, " has 0 instances of subject speech")
            try:
                ti_td_speech = durations[(durations['type'] == 'speech') & (durations['speaker'] == 'Interviewer')]['duration'].values[0]
            except IndexError:
                ti_td_speech = 0
                print("WARNING: ", file, " has 0 instances of interviewer speech")
            try:
                to_td_speech = durations[(durations['type'] == 'speech') & (durations['speaker'] == 'Other')]['duration'].values[0]
            except IndexError:
                to_td_speech = 0
            try:
                tsk_td_speech = durations[(durations['type'] == 'speech') & (durations['speaker'] == 'SKIP')]['duration'].values[0]
            except IndexError:
                tsk_td_speech = 0
            tb_td_speech = durations[((durations['type'] == 'speech') & (durations['speaker'] == 'Interviewer+Subject')) |
                                        ((durations['type'] == 'speech') & (durations['speaker'] == 'Subject+Interviewer'))]['duration'].sum()

            try:
                t_n_speech = counts[(counts['type'] == 'speech') & (counts['speaker'] == 'Subject')]['count'].values[0]
            except IndexError:
                t_n_speech = 0
                print("WARNING: ", file, " has 0 instances of subject speech")
            try:
                ti_n_speech = counts[(counts['type'] == 'speech') & (counts['speaker'] == 'Interviewer')]['count'].values[0]
            except IndexError:
                ti_n_speech = 0
                print("WARNING: ", file, " has 0 instances of interviewer speech")
            try:
                to_n_speech = counts[(counts['type'] == 'speech') & (counts['speaker'] == 'Other')]['count'].values[0]
            except IndexError:
                to_n_speech = 0
            try:
                tsk_n_speech = counts[(counts['type'] == 'speech') & (counts['speaker'] == 'SKIP')]['count'].values[0]
            except IndexError:
                tsk_n_speech = 0
            tb_n_speech = counts[((counts['type'] == 'speech') & (counts['speaker'] == 'Interviewer+Subject')) |
                                    ((counts['type'] == 'speech') & (counts['speaker'] == 'Subject+Interviewer'))]['count'].sum()



            # Latency Features
            try:
                t_td_latency = durations[(durations['type'] == 'latency') & (durations['speaker'] == 'Subject')]['duration'].values[0]
            except IndexError:
                t_td_latency = 0
                print("WARNING: ", file, " has 0 instances of subject latency")
            try:
                ti_td_latency = durations[(durations['type'] == 'latency') & (durations['speaker'] == 'Interviewer')]['duration'].values[0]
            except IndexError:
                ti_td_latency = 0
                print("WARNING: ", file, " has 0 instances of interviewer latency")
            try:
                to_td_latency = durations[(durations['type'] == 'latency') & (durations['speaker'] == 'Other')]['duration'].values[0]
            except IndexError:
                to_td_latency = 0
            try:
                tsk_td_latency = durations[(durations['type'] == 'latency') & (durations['speaker'] == 'SKIP')]['duration'].values[0]
            except IndexError:
                tsk_td_latency = 0
            try:
                tb_td_latency = durations[((durations['type'] == 'latency') & (durations['speaker'] == 'Interviewer+Subject')) |
                                     ((durations['type'] == 'latency') & (durations['speaker'] == 'Subject+Interviewer'))]['duration'].sum()
            except IndexError:
                tb_td_latency = 0

            try:
                t_n_latency = counts[(counts['type'] == 'latency') & (counts['speaker'] == 'Subject')]['count'].values[0]
            except IndexError:
                t_n_latency = 0
                print("WARNING: ", file, " has 0 instances of subject latency")
            try:
                ti_n_latency = counts[(counts['type'] == 'latency') & (counts['speaker'] == 'Interviewer')]['count'].values[0]
            except IndexError:
                ti_n_latency = 0
                print("WARNING: ", file, " has 0 instances of interviewer latency")
            try:
                to_n_latency = counts[(counts['type'] == 'latency') & (counts['speaker'] == 'Other')]['count'].values[0]
            except IndexError:
                to_n_latency = 0
            try:
                tsk_n_latency = counts[(counts['type'] == 'latency') & (counts['speaker'] == 'SKIP')]['count'].values[0]
            except IndexError:
                tsk_n_latency = 0
            try:
                tb_n_latency = counts[((counts['type'] == 'latency') & (counts['speaker'] == 'Interviewer+Subject')) |
                                     ((counts['type'] == 'latency') & (counts['speaker'] == 'Subject+Interviewer'))]['count'].sum()
            except IndexError:
                tb_n_latency = 0



            # Pause Features
            try:
                t_td_pause = durations[(durations['type'] == 'pause') & (durations['speaker'] == 'Subject')]['duration'].values[0]
            except IndexError:
                t_td_pause = 0
                print("WARNING: ", file, " has 0 instances of subject pause")
            try:
                ti_td_pause = durations[(durations['type'] == 'pause') & (durations['speaker'] == 'Interviewer')]['duration'].values[0]
            except IndexError:
                ti_td_pause = 0
                print("WARNING: ", file, " has 0 instances of interviewer pause")
            try:
                to_td_pause = durations[(durations['type'] == 'pause') & (durations['speaker'] == 'Other')]['duration'].values[0]
            except IndexError:
                to_td_pause = 0
            try:
                tsk_td_pause = durations[(durations['type'] == 'pause') & (durations['speaker'] == 'SKIP')]['duration'].values[0]
            except IndexError:
                tsk_td_pause = 0
            try:
                tb_td_pause = durations[((durations['type'] == 'pause') & (durations['speaker'] == 'Interviewer+Subject')) |
                                     ((durations['type'] == 'pause') & (durations['speaker'] == 'Subject+Interviewer'))]['duration'].sum()
            except IndexError:
                tb_td_pause = 0

            try:
                t_n_pause = counts[(counts['type'] == 'pause') & (counts['speaker'] == 'Subject')]['count'].values[0]
            except IndexError:
                t_n_pause = 0
                print("WARNING: ", file, " has 0 instances of subject pause")
            try:
                ti_n_pause = counts[(counts['type'] == 'pause') & (counts['speaker'] == 'Interviewer')]['count'].values[0]
            except IndexError:
                ti_n_pause = 0
                print("WARNING: ", file, " has 0 instances of interviewer pause")
            try:
                to_n_pause = counts[(counts['type'] == 'pause') & (counts['speaker'] == 'Other')]['count'].values[0]
            except IndexError:
                to_n_pause = 0
            try:
                tsk_n_pause = counts[(counts['type'] == 'pause') & (counts['speaker'] == 'SKIP')]['count'].values[0]
            except IndexError:
                tsk_n_pause = 0
            try:
                tb_n_pause = counts[((counts['type'] == 'pause') & (counts['speaker'] == 'Interviewer+Subject')) |
                                     ((counts['type'] == 'pause') & (counts['speaker'] == 'Subject+Interviewer'))]['count'].sum()
            except IndexError:
                tb_n_pause = 0


            # Median, Mean, 95th Percentile for speech, latency, and pause
            # Group by speech-type, speaker, and calculate median, mean, and 95th percentile duration
            result = []
            for typ in ['speech', 'pause', 'latency', 'NA_SKIP', 'NA_SPEAKER']:
                for speaker, group in labs_df[labs_df['type'] == typ].groupby('speaker'):
                    median_duration = group['duration'].median()
                    mean_duration = group['duration'].mean()
                    p95_duration = group['duration'].quantile(0.95)
                    min_duration = group['duration'].min()
                    max_duration = group['duration'].max()
                    result.append({
                        'type': typ,
                        'speaker': speaker,
                        'median_duration': median_duration,
                        'mean_duration': mean_duration,
                        'p95_duration': p95_duration,
                        'min_duration': min_duration,
                        'max_duration': max_duration
                    })

            # Convert result to DataFrame and assign to variables
            result_df = pd.DataFrame(result)

            # Speech Measures of Central Tendency
            try:
                t_md_speech = result_df[(result_df['type'] == 'speech') & (result_df['speaker'] == 'Subject')]['median_duration'].values[0]
                t_mn_speech = result_df[(result_df['type'] == 'speech') & (result_df['speaker'] == 'Subject')]['mean_duration'].values[0]
                t_95_speech = result_df[(result_df['type'] == 'speech') & (result_df['speaker'] == 'Subject')]['p95_duration'].values[0]
                t_min_speech = result_df[(result_df['type'] == 'speech') & (result_df['speaker'] == 'Subject')]['min_duration'].values[0]
                t_mx_speech = result_df[(result_df['type'] == 'speech') & (result_df['speaker'] == 'Subject')]['max_duration'].values[0]
            except IndexError:
                t_md_speech = 0
                t_mn_speech = 0
                t_95_speech = 0
                t_min_speech = 0
                t_mx_speech = 0
            try:
                ti_md_speech = result_df[(result_df['type'] == 'speech') & (result_df['speaker'] == 'Interviewer')]['median_duration'].values[0]
                ti_mn_speech = result_df[(result_df['type'] == 'speech') & (result_df['speaker'] == 'Interviewer')]['mean_duration'].values[0]
                ti_95_speech = result_df[(result_df['type'] == 'speech') & (result_df['speaker'] == 'Interviewer')]['p95_duration'].values[0]
                ti_min_speech = result_df[(result_df['type'] == 'speech') & (result_df['speaker'] == 'Interviewer')]['min_duration'].values[0]
                ti_mx_speech = result_df[(result_df['type'] == 'speech') & (result_df['speaker'] == 'Interviewer')]['max_duration'].values[0]
            except IndexError:
                ti_md_speech = 0
                ti_mn_speech = 0
                ti_95_speech = 0
                ti_min_speech = 0
                ti_mx_speech = 0
            try:
                to_md_speech = result_df[(result_df['type'] == 'speech') & (result_df['speaker'] == 'Other')]['median_duration'].values[0]
                to_mn_speech = result_df[(result_df['type'] == 'speech') & (result_df['speaker'] == 'Other')]['mean_duration'].values[0]
                to_95_speech = result_df[(result_df['type'] == 'speech') & (result_df['speaker'] == 'Other')]['p95_duration'].values[0]
                to_min_speech = result_df[(result_df['type'] == 'speech') & (result_df['speaker'] == 'Other')]['min_duration'].values[0]
                to_mx_speech = result_df[(result_df['type'] == 'speech') & (result_df['speaker'] == 'Other')]['max_duration'].values[0]
            except IndexError:
                to_md_speech = 0
                to_mn_speech = 0
                to_95_speech = 0
                to_min_speech = 0
                to_mx_speech = 0
            try:
                tb_md_speech = result_df[((result_df['type'] == 'speech') & (result_df['speaker'] == 'Interviewer+Subject')) |
                                         ((result_df['type'] == 'speech') & (result_df['speaker'] == 'Subject+Interviewer'))]['median_duration'].sum()
                tb_mn_speech = result_df[((result_df['type'] == 'speech') & (result_df['speaker'] == 'Interviewer+Subject')) |
                                         ((result_df['type'] == 'speech') & (result_df['speaker'] == 'Subject+Interviewer'))]['mean_duration'].sum()
                tb_95_speech = result_df[((result_df['type'] == 'speech') & (result_df['speaker'] == 'Interviewer+Subject')) |
                                         ((result_df['type'] == 'speech') & (result_df['speaker'] == 'Subject+Interviewer'))]['p95_duration'].sum()
                tb_min_speech = result_df[((result_df['type'] == 'speech') & (result_df['speaker'] == 'Interviewer+Subject')) |
                                         ((result_df['type'] == 'speech') & (result_df['speaker'] == 'Subject+Interviewer'))]['min_duration'].sum()
                tb_mx_speech = result_df[((result_df['type'] == 'speech') & (result_df['speaker'] == 'Interviewer+Subject')) |
                                         ((result_df['type'] == 'speech') & (result_df['speaker'] == 'Subject+Interviewer'))]['max_duration'].sum()
            except IndexError:
                tb_md_speech = 0
                tb_mn_speech = 0
                tb_95_speech = 0
                tb_min_speech = 0
                tb_mx_speech = 0


            # Latency Measures of Central Tendency
            try:
                t_md_latency = result_df[(result_df['type'] == 'latency') & (result_df['speaker'] == 'Subject')]['median_duration'].values[0]
                t_mn_latency = result_df[(result_df['type'] == 'latency') & (result_df['speaker'] == 'Subject')]['mean_duration'].values[0]
                t_95_latency = result_df[(result_df['type'] == 'latency') & (result_df['speaker'] == 'Subject')]['p95_duration'].values[0]
                t_min_latency = result_df[(result_df['type'] == 'latency') & (result_df['speaker'] == 'Subject')]['min_duration'].values[0]
                t_mx_latency = result_df[(result_df['type'] == 'latency') & (result_df['speaker'] == 'Subject')]['max_duration'].values[0]
            except IndexError:
                t_md_latency = 0
                t_mn_latency = 0
                t_95_latency = 0
                t_min_latency = 0
                t_mx_latency = 0
            try:
                ti_md_latency = result_df[(result_df['type'] == 'latency') & (result_df['speaker'] == 'Interviewer')]['median_duration'].values[0]
                ti_mn_latency = result_df[(result_df['type'] == 'latency') & (result_df['speaker'] == 'Interviewer')]['mean_duration'].values[0]
                ti_95_latency = result_df[(result_df['type'] == 'latency') & (result_df['speaker'] == 'Interviewer')]['p95_duration'].values[0]
                ti_min_latency = result_df[(result_df['type'] == 'latency') & (result_df['speaker'] == 'Interviewer')]['min_duration'].values[0]
                ti_mx_latency = result_df[(result_df['type'] == 'latency') & (result_df['speaker'] == 'Interviewer')]['max_duration'].values[0]
            except IndexError:
                ti_md_latency = 0
                ti_mn_latency = 0
                ti_95_latency = 0
                ti_min_latency = 0
                ti_mx_latency = 0
            try:
                to_md_latency = result_df[(result_df['type'] == 'latency') & (result_df['speaker'] == 'Other')]['median_duration'].values[0]
                to_mn_latency = result_df[(result_df['type'] == 'latency') & (result_df['speaker'] == 'Other')]['mean_duration'].values[0]
                to_95_latency = result_df[(result_df['type'] == 'latency') & (result_df['speaker'] == 'Other')]['p95_duration'].values[0]
                to_min_latency = result_df[(result_df['type'] == 'latency') & (result_df['speaker'] == 'Other')]['min_duration'].values[0]
                to_mx_latency = result_df[(result_df['type'] == 'latency') & (result_df['speaker'] == 'Other')]['max_duration'].values[0]
            except IndexError:
                to_md_latency = 0
                to_mn_latency = 0
                to_95_latency = 0
                to_min_latency = 0
                to_mx_latency = 0
            try:
                tb_md_latency = result_df[((result_df['type'] == 'latency') & (result_df['speaker'] == 'Interviewer+Subject')) |
                                         ((result_df['type'] == 'latency') & (result_df['speaker'] == 'Subject+Interviewer'))]['median_duration'].sum()
                tb_mn_latency = result_df[((result_df['type'] == 'latency') & (result_df['speaker'] == 'Interviewer+Subject')) |
                                         ((result_df['type'] == 'latency') & (result_df['speaker'] == 'Subject+Interviewer'))]['mean_duration'].sum()
                tb_95_latency = result_df[((result_df['type'] == 'latency') & (result_df['speaker'] == 'Interviewer+Subject')) |
                                         ((result_df['type'] == 'latency') & (result_df['speaker'] == 'Subject+Interviewer'))]['p95_duration'].sum()
                tb_min_latency = result_df[((result_df['type'] == 'latency') & (result_df['speaker'] == 'Interviewer+Subject')) |
                                         ((result_df['type'] == 'latency') & (result_df['speaker'] == 'Subject+Interviewer'))]['min_duration'].sum()
                tb_mx_latency = result_df[((result_df['type'] == 'latency') & (result_df['speaker'] == 'Interviewer+Subject')) |
                                         ((result_df['type'] == 'latency') & (result_df['speaker'] == 'Subject+Interviewer'))]['max_duration'].sum()
            except IndexError:
                tb_md_latency = 0
                tb_mn_latency = 0
                tb_95_latency = 0
                tb_min_latency = 0
                tb_mx_latency= 0

            # Pause Measures of Central Tendency
            try:
                t_md_pause = result_df[(result_df['type'] == 'pause') & (result_df['speaker'] == 'Subject')]['median_duration'].values[0]
                t_mn_pause = result_df[(result_df['type'] == 'pause') & (result_df['speaker'] == 'Subject')]['mean_duration'].values[0]
                t_95_pause = result_df[(result_df['type'] == 'pause') & (result_df['speaker'] == 'Subject')]['p95_duration'].values[0]
                t_min_pause = result_df[(result_df['type'] == 'pause') & (result_df['speaker'] == 'Subject')]['min_duration'].values[0]
                t_mx_pause = result_df[(result_df['type'] == 'pause') & (result_df['speaker'] == 'Subject')]['max_duration'].values[0]
            except IndexError:
                t_md_pause = 0
                t_mn_pause = 0
                t_95_pause = 0
                t_min_pause = 0
                t_mx_pause = 0
            try:
                ti_md_pause = result_df[(result_df['type'] == 'pause') & (result_df['speaker'] == 'Interviewer')]['median_duration'].values[0]
                ti_mn_pause = result_df[(result_df['type'] == 'pause') & (result_df['speaker'] == 'Interviewer')]['mean_duration'].values[0]
                ti_95_pause = result_df[(result_df['type'] == 'pause') & (result_df['speaker'] == 'Interviewer')]['p95_duration'].values[0]
                ti_min_pause = result_df[(result_df['type'] == 'pause') & (result_df['speaker'] == 'Interviewer')]['min_duration'].values[0]
                ti_mx_pause = result_df[(result_df['type'] == 'pause') & (result_df['speaker'] == 'Interviewer')]['max_duration'].values[0]
            except IndexError:
                ti_md_pause = 0
                ti_mn_pause = 0
                ti_95_pause = 0
                ti_min_pause = 0
                ti_mx_pause = 0
            try:
                to_md_pause = result_df[(result_df['type'] == 'pause') & (result_df['speaker'] == 'Other')]['median_duration'].values[0]
                to_mn_pause = result_df[(result_df['type'] == 'pause') & (result_df['speaker'] == 'Other')]['mean_duration'].values[0]
                to_95_pause = result_df[(result_df['type'] == 'pause') & (result_df['speaker'] == 'Other')]['p95_duration'].values[0]
                to_min_pause = result_df[(result_df['type'] == 'pause') & (result_df['speaker'] == 'Other')]['min_duration'].values[0]
                to_mx_pause = result_df[(result_df['type'] == 'pause') & (result_df['speaker'] == 'Other')]['max_duration'].values[0]
            except IndexError:
                to_md_pause = 0
                to_mn_pause = 0
                to_95_pause = 0
                to_min_pause = 0
                to_mx_pause = 0
            try:
                tb_md_pause = result_df[((result_df['type'] == 'pause') & (result_df['speaker'] == 'Interviewer+Subject')) |
                                         ((result_df['type'] == 'pause') & (result_df['speaker'] == 'Subject+Interviewer'))]['median_duration'].sum()
                tb_mn_pause = result_df[((result_df['type'] == 'pause') & (result_df['speaker'] == 'Interviewer+Subject')) |
                                         ((result_df['type'] == 'pause') & (result_df['speaker'] == 'Subject+Interviewer'))]['mean_duration'].sum()
                tb_95_pause = result_df[((result_df['type'] == 'pause') & (result_df['speaker'] == 'Interviewer+Subject')) |
                                         ((result_df['type'] == 'pause') & (result_df['speaker'] == 'Subject+Interviewer'))]['p95_duration'].sum()
                tb_min_pause = result_df[((result_df['type'] == 'pause') & (result_df['speaker'] == 'Interviewer+Subject')) |
                                         ((result_df['type'] == 'pause') & (result_df['speaker'] == 'Subject+Interviewer'))]['min_duration'].sum()
                tb_mx_pause = result_df[((result_df['type'] == 'pause') & (result_df['speaker'] == 'Interviewer+Subject')) |
                                         ((result_df['type'] == 'pause') & (result_df['speaker'] == 'Subject+Interviewer'))]['max_duration'].sum()
            except IndexError:
                tb_md_pause = 0
                tb_mn_pause = 0
                tb_95_pause = 0
                tb_min_pause = 0
                tb_mx_pause = 0


            # Pause Duration Counts: short [0.25s-1s), medium [1s - 2s), long [2s+]
            # Filter 'non-speech' segments
            pause_df = labs_df[labs_df['type'] == 'pause']

            # Define bins for durations
            bins = [0.25, 1, 2, float('inf')]
            labels = ['[250ms - 1s)', '[1s - 2s)', '[2s+)']

            # Group by speaker and count durations in each bin
            result_pause = []
            for speaker, group in pause_df.groupby('speaker'):
                counts = pd.cut(group['duration'], bins=bins, labels=labels, include_lowest=True, right=False).value_counts().sort_index().values
                result_pause.append({
                    'speaker': speaker,
                    '[250ms - 1s)': counts[0] if len(counts) > 0 else 0,
                    '[1s - 2s)': counts[1] if len(counts) > 1 else 0,
                    '[2s+)': counts[2] if len(counts) > 2 else 0
                })

            # Convert result to DataFrame
            result_pause_df = pd.DataFrame(result_pause)

            try:
                t_n_pauseshort = result_pause_df[result_pause_df['speaker'] == 'Subject']['[250ms - 1s)'].values[0]
                t_n_pausemed = result_pause_df[result_pause_df['speaker'] == 'Subject']['[1s - 2s)'].values[0]
                t_n_pauselong = result_pause_df[result_pause_df['speaker'] == 'Subject']['[2s+)'].values[0]
            except (IndexError, KeyError) as e:
                t_n_pauseshort = 0
                t_n_pausemed = 0
                t_n_pauselong = 0
            try:
                ti_n_pauseshort = result_pause_df[result_pause_df['speaker'] == 'Interviewer']['[250ms - 1s)'].values[0]
                ti_n_pausemed = result_pause_df[result_pause_df['speaker'] == 'Interviewer']['[1s - 2s)'].values[0]
                ti_n_pauselong = result_pause_df[result_pause_df['speaker'] == 'Interviewer']['[2s+)'].values[0]
            except (IndexError, KeyError) as e:
                ti_n_pauseshort = 0
                ti_n_pausemed = 0
                ti_n_pauselong = 0



            # Overall Features
            t_td_speaker = t_td_speech + t_td_latency + t_td_pause
            ti_td_speaker = ti_td_speech + ti_td_latency + ti_td_pause
            to_td_speaker = to_td_speech + to_td_latency + to_td_pause
            tsk_td_speaker = tsk_td_speech + tsk_td_latency + tsk_td_pause
            tb_td_speaker = tb_td_speech + tb_td_latency + tb_td_pause

            t_td_task = t_td_speaker + ti_td_speaker + to_td_speaker + tb_td_speaker


            # Proportional Features
            total_speech_time = t_td_speech + ti_td_speech + tb_td_speech
            t_pd_speech = (t_td_speech + tb_td_speech) / total_speech_time
            ti_pd_speech = (ti_td_speech + tb_td_speech) / total_speech_time
            to_pd_speech = (to_td_speech) / total_speech_time
            tsk_pd_speech = (tsk_td_speech) / total_speech_time
            tb_pd_speech = (tb_td_speech) / total_speech_time

            t_pd_pause = (t_td_pause + tb_td_pause) / total_speech_time
            ti_pd_pause = (ti_td_pause + tb_td_pause) / total_speech_time
            to_pd_pause = (to_td_pause) / total_speech_time
            tsk_pd_pause = (tsk_td_pause) / total_speech_time
            tb_pd_pause = (tb_td_pause) / total_speech_time

            t_pd_latency = (t_td_latency + tb_td_latency) / total_speech_time
            ti_pd_latency = (ti_td_latency + tb_td_latency) / total_speech_time
            to_pd_latency = (to_td_latency) / total_speech_time
            tsk_pd_latency = (tsk_td_latency) / total_speech_time
            tb_pd_latency = (tb_td_latency) / total_speech_time


            features.append([filename, study_name, user_id, timepoint, sample_id, version, task, stimulus,
                            t_td_task, t_td_speaker, ti_td_speaker, to_td_speaker, tsk_td_speaker, tb_td_speaker,
                            t_td_speech, ti_td_speech, to_td_speech, tsk_td_speech, tb_td_speech,
                            total_speech_time, t_pd_speech, ti_pd_speech, to_pd_speech, tsk_pd_speech, tb_pd_speech,
                            t_td_latency, ti_td_latency, to_td_latency, tsk_td_latency, tb_td_latency,
                            t_pd_latency, ti_pd_latency, to_pd_latency, tsk_pd_latency, tb_pd_latency,
                            t_td_pause, ti_td_pause, to_td_pause, tsk_td_pause, tb_td_pause,
                            t_pd_pause, ti_pd_pause, to_pd_pause, tsk_pd_pause, tb_pd_pause,
                            t_n_speech, ti_n_speech, to_n_speech, tsk_n_speech, tb_n_speech,
                            t_n_latency, ti_n_latency, to_n_latency, tsk_n_latency, tb_n_latency,
                            t_n_pause, ti_n_pause, to_n_pause, tsk_n_pause, tb_n_pause,
                            t_n_pauseshort, t_n_pausemed, t_n_pauselong, ti_n_pauseshort, ti_n_pausemed, ti_n_pauselong,
                            t_md_speech, t_mn_speech, t_95_speech, t_min_speech, t_mx_speech,
                            ti_md_speech, ti_mn_speech, ti_95_speech, ti_min_speech, ti_mx_speech,
                            to_md_speech, to_mn_speech, to_95_speech, to_min_speech, to_mx_speech,
                            tb_md_speech, tb_mn_speech, tb_95_speech, tb_min_speech, tb_mx_speech,
                            t_md_pause, t_mn_pause, t_95_pause, t_min_pause, t_mx_pause,
                            ti_md_pause, ti_mn_pause, ti_95_pause, ti_min_pause, ti_mx_pause,
                            to_md_pause, to_mn_pause, to_95_pause, to_min_pause, to_mx_pause,
                            tb_md_pause, tb_mn_pause, tb_95_pause, tb_min_pause, tb_mx_pause,
                            t_md_latency, t_mn_latency, t_95_latency, t_min_latency, t_mx_latency,
                            ti_md_latency, ti_mn_latency, ti_95_latency, ti_min_latency, ti_mx_latency,
                            to_md_latency, to_mn_latency, to_95_latency, to_min_latency, to_mx_latency,
                            tb_md_latency, tb_mn_latency, tb_95_latency, tb_min_latency, tb_mx_latency])

        # Ignore header of list
        features_df = pd.DataFrame(features[1:], columns=headers)

        all_features_df = features_df[all_features]
        primary_features_df = features_df[primary_features]
        supp_features_df = features_df[supp_features]

        all_features_name = self.feat_dir / 'timing_features_all.tsv'
        primary_features_name = self.feat_dir / 'timing_features_primary.tsv'
        supp_features_name = self.feat_dir / 'timing_features_supp.tsv'

        all_features_df.to_csv(all_features_name, sep='\t', index=False)
        primary_features_df.to_csv(primary_features_name, sep='\t', index=False)
        supp_features_df.to_csv(supp_features_name, sep='\t', index=False)

        print("FINISHED!")


'''# Capture stdout and stderr
log_out_name = features_path + '/' + 'log_stdout_features.txt'
log_err_name = features_path + '/' + 'log_stderr_features.txt'
with open(log_out_name, 'w', newline='') as f:
    f.write(cap.stdout)
with open(log_err_name, 'w', newline='') as f:
    f.write(cap.stderr)
'''

# Statistical Analysis for Labeled SAD Lab files
#
# Core Features
# * Overall features
#     * t_td_task - total duration of the audio for the whole task, including all speakers
#     * t_td_speaker - total duration of the audio identified to that speaker, sum of all pauses, latencies, and speaking time
# * Latency features
#     * t_td_latency, t_pd_latency - total duration of latencies (for that speaker), and duration of the latency as a proportion of the total speaker time
#     * t_n_latency - total number of latencies
#     * t_md_latency, t_mx_latency - median and 95% (max) latency in milliseconds
# * Pause Features
#     * t_td_pause, t_pd_pause - total duration of pauses (for that speaker), and duration of the pauses as a proportion of the total speaker time
#     * t_n_pause - total number of any pause (>250ms)
#     * t_md_pauselength, t_mx_pauselength - median and max pause duration in milliseconds
#     * t_n_pauseshort, t_n_pausemed, t_n_pauselong - total number of short pauses [250ms - 1s), medium pauses [1s - 2s), and long pauses [2s+].
# * Speaking Time Features
#     * t_td_speech, t_pd_speech - total duration of speaking time for the speaker; proportional duration, normalized by total speaker time
#     * t_n_speech - number of speech segments
#     * t_mi_speech, t_md_speech, t_mx_speech - min, median, and max speech segment duration (as punctuated by pauses or other speaker)
# * Features derived from acoustic + word level - for later:
#     * speaking rate - words per time
#     * articulation rate - syllable count per second
# * THINGS TO CHECK: does td_latency + td_pause + td_speak = td_speaker? does pd_latency + pd_pause + pd_speak = 1?
#
#
#
#
# Supplementary Features
# * Overall features
#     * (DONE) t_td_task - total duration of the audio for the whole task, including all speakers
#     * (DONE) t_td_speaker - total duration of the audio identified to that speaker, sum of all pauses, latencies, and speaking time
# * Latency features
#     * (DONE) t_td_latency, (TODO) t_pd_latency - total duration of latencies (for that speaker), and duration of the latency as a proportion of the total speaker time
#     * (DONE) t_n_latency - total number of latencies
#     * (DONE) t_md_latency, t_mn_latency, t_95_latency, t_min_latency, t_mx_latency - median, mean, 95% percentile, minimum and max latency duration in seconds
# * Pause Features
#     * (DONE) t_td_pause, (TODO) t_pd_pause - total duration of pauses (for that speaker), and duration of the pauses as a proportion of the total speaker time
#     * (DONE) t_n_pause - total number of any pause (>250ms)
#     * (DONE) t_md_pause, t_mn_pause, t_95_pause, t_min_pause, t_mx_pause - median, mean, 95th percentile, minimum, and max pause duration in seconds
#     * (DONE) t_n_pauseshort, t_n_pausemed, t_n_pauselong - total number of short pauses [250ms - 1s), medium pauses [1s - 2s), and long pauses [2s+].
# * Speaking Time Features
#     * (DONE) t_td_speech, (TODO) t_pd_speech - total duration of speaking time for the speaker; proportional duration, normalized by total speaker time
#     * (DONE) t_n_speech - number of speech segments
#     * (DONE) t_md_speech, t_mn_speech, t_95_speech t_min_speech, t_mx_speech -  median, mean, 95th percentile, and max speech segment duration (as punctuated by pauses or other speaker)
#
#
# NOTE: for above features -  "t_" for Subject
#                             "ti_" for Interviewer
#                             "to_" for Other
#                             "tsk_" for Skip
#                             "tb" for both Interviewer and Subject Speaking
#
#     Durations are in seconds
#
#
#
# /Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/sad_out/remora/post_labs/remora_13394_T1_13394T142_v2023_PIC_Rorschach8_LABELED.lab  is being processed...
# Durations:
#        speaker     type  duration
# 0  Interviewer  latency      2.21
# 1  Interviewer   speech      3.37
# 2      Subject  latency      0.50
# 3      Subject    pause      1.11
# 4      Subject   speech      9.54
#
#
# durations:
#        speaker     type  count
# 0  Interviewer  latency      1
# 1  Interviewer   speech      2
# 2      Subject  latency      1
# 3      Subject    pause      3
# 4      Subject   speech      4
#
#
#
# /Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/sad_out/remora/post_labs/remora_13413_T1_13413T101_v2023_BTW_BTW_LABELED.lab  is being processed...
# Durations:
#                 speaker     type  duration
# 0           Interviewer  NA_SKIP     4.210
# 1           Interviewer  latency     3.000
# 2           Interviewer    pause    10.290
# 3           Interviewer   speech    64.594
# 4   Interviewer+Subject   speech     0.515
# 5                  SKIP  latency     1.830
# 6                  SKIP    pause   151.310
# 7                  SKIP   speech   246.860
# 8               Subject  latency     4.620
# 9               Subject    pause    50.850
# 10              Subject   speech    35.161
# 11  Subject+Interviewer   speech     0.644
#
#
# durations:
#                 speaker     type  count
# 0           Interviewer  NA_SKIP      2
# 1           Interviewer  latency      4
# 2           Interviewer    pause      9
# 3           Interviewer   speech     18
# 4   Interviewer+Subject   speech      1
# 5                  SKIP  latency      2
# 6                  SKIP    pause    117
# 7                  SKIP   speech    119
# 8               Subject  latency      5
# 9               Subject    pause     23
# 10              Subject   speech     30
# 11  Subject+Interviewer   speech      1

# In[16]:

