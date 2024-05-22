#!/usr/bin/env python
# coding: utf-8

# # torch==2.2.1
# # noisereduce==3.0.2
# # librosa==0.10.1
# # soundfile==0.12.1

# In[2]:


import numpy as np
import pandas as pd
from scipy.io import wavfile
from pathlib import Path

import soundfile as sf
import noisereduce as nr
import librosa


def run_SpeechQuality(file):
    windowT = 0.025
    incrT = 0.01
    X, FS = librosa.load(file)
    # FS, X = wavfile.read(file)
    windowN = round(windowT*FS)
    incrN = round(incrT*FS)
    if X.ndim == 2:  # If the file is stereo
        # X = X.flatten() # Flatten and measure for both. (average?)
        X = X.mean(axis=1)  # Average the two channels (replace later with absmax)
    H = np.hamming(windowN)
    nsamples = len(X)
    lastsamp = nsamples-windowN
    nrms = len(np.arange(0, lastsamp, incrN))
    MS = np.zeros((nrms, 1))
    count = 0
    for s in np.arange(0, lastsamp, incrN):
        # print(X[s:min(len(X), (s + windowN))].size)
        # print(H.size)
        Sig = X[s:min(len(X), (s + windowN))] * H
        MS[count] = (np.conj(Sig) @ Sig) / windowN
        count += 1
    Q15 = 10 * np.log10(np.quantile(MS, 0.15))
    Q85 = 10 * np.log10(np.quantile(MS, 0.85))
    aX = abs(X)
    nclipped = sum(aX >= 1)
    return Q85 - Q15, nclipped


def denoise(audio_full_path, denoise_level=0.90):
    # Load the audio file using librosa
    audio_data, sample_rate = librosa.load(audio_full_path, mono=False, sr=None)
    orig_shape = audio_data.shape
    audio_data = audio_data.flatten()
    two_channel = len(orig_shape) > 1 and orig_shape[1] >= 2
    if two_channel:  # If two channels, flatten and process, then reshape later.
        audio_data = audio_data.flatten()
    # denoise the audio
    sample_rate = int(sample_rate)
    reduced_audio = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=denoise_level)
    if two_channel:
        reduced_audio = np.reshape(reduced_audio, orig_shape).T
    # print(reduced_audio.shape)
    return reduced_audio, sample_rate

class SNRMeasurer:

    def __init__(self, recording_path, output_path, denoised_path, interim_csvs=True):
        self.interim_csvs = interim_csvs  # Whether to save as csv at each intermediate step or keep dataframes in memory.
        self.recording_path = recording_path
        self.output_path = output_path
        self.denoised_path = denoised_path
        # self.all_audio_files = list(self.recording_path.glob("*.wav")) + list(self.recording_path.glob("*.mp4"))

    def make_snr_df(self, path_to_audios, name="base_snr.csv", columns=None):
        # Generate SNRs and nclipped for each file and save in pandas df
        # This first is used to identify whether noise reduction should be performed.
        # Then, it is used on the files that are denoised to obtain the final values.
        df_list = [pd.DataFrame(columns=columns)]
        files = (p.resolve() for p in Path(path_to_audios).glob("**/*") if p.suffix in {".mp4", ".wav", ".mp3", ".m4a"})
        for each_recording in files:
            print(each_recording)
            each_entry = [each_recording]
            snr_tupple = run_SpeechQuality(each_recording)
            each_entry.append(snr_tupple[0])
            each_entry.append(snr_tupple[1])
            each_entry = pd.DataFrame([each_entry], columns=columns)
            df_list.append(each_entry)
        print(df_list)
        df_snr = pd.concat(df_list, ignore_index=True)
        if self.interim_csvs:
            df_snr.to_csv(self.denoised_path / name, index=False)
        return df_snr

    def get_excl_incl(self, df_snr, snr_cutoff=15):
        # Masks results in the base snr df according to whether they are less than the snr cutoff.
        # Returns the excluded and included values in separate dfs.
        mask_cutoff = df_snr['SNR'] < snr_cutoff
        df_incl = df_snr[~mask_cutoff]
        df_excl = df_snr[mask_cutoff]
        if self.interim_csvs:
            df_excl.to_csv(self.denoised_path / 'snr_EXCLUDED.csv', index=False)
            df_incl.to_csv(self.denoised_path / 'snr_INCLUDED.csv', index=False)
        return df_excl, df_incl


    def get_merged_denoise_df(self, df_excl):
        # Run denoiser on all excluded files (files are excluded for low SNR)
        # We don't want to lose too much data in this process,
        # especially if you intend to take the denoised files for acoustic processing.
        for file_name in df_excl['file_name']:
            print(f"Running Denoiser On: {file_name}")
            reduced_audio, sample_rate = denoise(file_name)
            sf.write(self.denoised_path / file_name.name, reduced_audio, sample_rate) # Save denoised file.

        # Compare denoised files with prior files
        # denoised_audio_files = self.denoised_path.glob("*")
        df_denoise = self.make_snr_df(self.denoised_path, name="snr_denoised.csv", columns=['file_name', 'SNR_denoised', 'nclipped_denoised'])

        # Making sure merge indices match
        df_excl.loc['file_name'] = df_excl['file_name'].apply(lambda s: Path(str(s)).name)
        df_denoise.loc['file_name'] = df_excl['file_name'].apply(lambda s: Path(str(s)).name)

        merged_denoise_df = pd.merge(df_excl, df_denoise, on="file_name", how="inner")
        merged_denoise_df['eligible'] = (merged_denoise_df['SNR_denoised'] >= 15)
        # Checks that denoising worked above signal threshold.
        if self.interim_csvs:
            merged_denoise_df.to_csv(self.denoised_path / "snr_denoised.csv", index=False)
        # print(merged_denoise_df.describe())
        return merged_denoise_df

    def get_final_snr_df(self, merged_denoise_df, df_incl):
        # Merge snr info about the included files and the files that have been denoised.
        # og_snr_df = pd.read_csv(os.path.join(output_path, "rem_batch1_snr_INCLUDED.csv"))  # Files with 15 <= SNR
        df_incl.loc['denoised'] = False
        df_incl.loc['eligible'] = True
        merged_denoise_df.loc['denoised'] = True
        final_snr_df = pd.concat([df_incl, merged_denoise_df])
        return final_snr_df

    def run_SNR_pipe(self, name="snr_final.csv"):
        # This function allows for interaction with the SNR object, and gives "pseudocode" in a sense for what the program does.
        df_snr = self.make_snr_df(self.recording_path, columns=['file_name', 'SNR', 'nclipped'])
        df_excl, df_incl = self.get_excl_incl(df_snr)  # Two dataframes giving the excluded + included files targetted for denoise.
        merged_denoise_df = self.get_merged_denoise_df(df_excl)  # Makes snr df for the denoised data and merges eligibles.
        final_snr_df = self.get_final_snr_df(merged_denoise_df, df_incl)
        final_snr_df.to_csv(self.output_path / name, index=False)
        return final_snr_df


# Save combine now eligible denoised files and originally passing files into single dataframe and output to CSV
# (mark whether or not file was denoised)




# In[10]:


###LPOP
#df_snr.to_csv('C:\\Users\\ANikzad\\Desktop\\Local_Pipeline\\Data\\Winterlight_LPOP\\Batch1\\Report\\lpop_batch1_snr.csv')


# In[11]:


###ACES
#df_snr.to_csv('C:\\Users\\ANikzad\\Desktop\\Local_Pipeline\\Data\\Winterlight_ACES\\Batch1\\Report\\aces_batch1_snr.csv')


# In[12]:


###ACES-MDD
#df_snr.to_csv('C:\\Users\\ANikzad\\Desktop\\Local_Pipeline\\Data\\Winterlight_ACES-MDD\\Batch1\\Report\\aces-mdd_batch1_snr.csv')


# Remora SNR Descriptive Stats and Overview

# In[7]:


# df_snr = pd.read_csv(os.path.join(output_path, 'rem_batch1_snr.csv'))


# In[8]:


# print("ALL SNRs:")
# print(df_snr.describe())
#
# inf_mask = df_snr['SNR'].isin([np.inf])
# df_snr_masked_inf = df_snr[~inf_mask]
# print("\nInfinite Masked SNRs:")
# print(df_snr_masked_inf.describe())


# In[9]:


# df_snr_masked_inf['SNR'].plot.hist(bins=12, alpha=0.5)


# In[10]:


# Create mask to ignore rows with SNR > 60 AND SNR < 15



# LPOP SNR Results

# In[18]:


#df_snr = pd.read_csv('C:\\Users\\ANikzad\\Desktop\\Local_Pipeline\\Data\\Winterlight_LPOP\\Batch1\\Report\\lpop_batch1_snr.csv').drop('Unnamed: 0', axis=1)


# In[19]:


'''
print('mean:', df_snr['SNR'].mean())
print('std:', df_snr['SNR'].std())
df_snr['SNR'].plot.hist(bins=12, alpha=0.5)
'''


# ACES SNR Results

# In[20]:


#df_snr = pd.read_csv('C:\\Users\\ANikzad\\Desktop\\Local_Pipeline\\Data\\Winterlight_ACES\\Batch1\\Report\\aces_batch1_snr.csv').drop('Unnamed: 0', axis=1)


# In[21]:


#df_snr.describe()


# In[22]:


#df_snr.loc[df_snr['SNR'] == np.inf]


# In[23]:


'''
df_snr = df_snr.loc[df_snr['SNR'] != np.inf]
print('mean:', df_snr['SNR'].mean())
print('std:', df_snr['SNR'].std())
df_snr['SNR'].plot.hist(bins=12, alpha=0.5)
'''


# ACES-MDD SNR Results

# In[24]:


'''
df_snr = pd.read_csv('C:\\Users\\ANikzad\\Desktop\\Local_Pipeline\\Data\\Winterlight_ACES-MDD\\Batch1\\Report\\aces-mdd_batch1_snr.csv').drop('Unnamed: 0', axis=1)
'''


# In[25]:


#df_snr.loc[df_snr['SNR'] == 'Failed']


# In[26]:


'''
df_snr = df_snr.loc[df_snr['SNR'] != 'Failed']
df_snr['SNR'] = df_snr['SNR'].astype(float)
df_snr['SNR'].describe()
'''


# In[27]:


'''
print('mean:', df_snr['SNR'].mean())
print('std:', df_snr['SNR'].std())
df_snr['SNR'].plot.hist(bins=12, alpha=0.5)
'''


# Use sample ID to match transcript with recording 
#     in the recording file name, the sample ID is the second number starts with 8...
