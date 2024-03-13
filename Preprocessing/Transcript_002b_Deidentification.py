#!/usr/bin/env python
# coding: utf-8

# ## Prep Dataframe

# In[1]:


import re
import csv
import glob
import pandas as pd
import numpy as np
import os


# # extract all the instances of squared brackets strings per line

# In[2]:


### Points to the folder that contains all data
root_data = "C://Users//ANikzad//Desktop//Local_Pipeline//Data//"

### Specify Source Folder
    # --- Remora
std_transcript_location = "Remora-2023//Batch-1//1_std_transcripts//"
spellchecked_transcript_location = "Remora-2023//Batch-1//1b_std_transcripts_spellchecked//"

### Specify Destination folder
    # --- Remora
deid_transcript_location = "Remora-2023//Batch-1//2_deid_transcripts//"

drive_trancript_in_path = root_data+spellchecked_transcript_location
drive_out_path = root_data + deid_transcript_location


# In[3]:


transcripts = os.listdir(drive_trancript_in_path)
len(transcripts)


# In[55]:


phi_df = pd.DataFrame()
for transcript in transcripts:
    df = pd.read_csv(drive_trancript_in_path + transcript, sep='\t', names=['stimulus', 'speaker','start_t','end_t','content'])
    #print(df['content'])

    for index, row in df.iterrows():
        content = row['content']
        
        phi = re.findall(r"\[([A-Za-z~',!:?.\- \s]+)\]",content)
        if len(phi) > 0:
            print(content)
            for each_phi in phi:
                phi_df = pd.concat([phi_df,pd.DataFrame([{'phi': each_phi}])], ignore_index=True)
            print(phi)
            print('\n')
phi_df


# In[37]:





# In[38]:


phi_df.drop_duplicates().reset_index().to_csv(root_data+'Remora-2023//Batch-1//reports//deid_report//remora_batch1_entity_set.csv')


# # Deidentify

# In[ ]:


'''workflow
df.head() to check

then copy ne_set outputs 
and paset it to an existing entity_csv file 
sep each col by ;
when pasteing, right click on cell [0,0], 
then select 'paste special'--'text'--'ok'
then clean up with 'replace'
Note: be careful with replacing ' ' with ''
Make sure the replacement is of the SAME length as the original
no header needed

then mannually checked each ne's context and make the replacement as meaningful as possible
then need to manually replace the first row's name csv
'''


# In[4]:


phi_df = pd.read_csv(root_data+'Remora-2023//Batch-1//reports//deid_report//MASTER_PHI.csv')
phi_dic = {}
for index, row in phi_df.iterrows():
    phi_dic[row['phi']] = row['replacement']
phi_dic


# In[23]:


for transcript in transcripts:
    df = pd.read_csv(drive_trancript_in_path + transcript, sep='\t', names=['stimulus', 'speaker','start_time','end_time','content'])
    #print(df['content'])
    df['speaker'].fillna('', inplace=True)
    with open(root_data + deid_transcript_location + transcript, 'w') as deid_transcript:
        for index, row in df.iterrows():
            content = row['content']
            phi = re.findall(r"\[([A-Za-z~',!:?.\- \s]+)\]",content)
            if len(phi) > 0:
                #print(content)
                for each_phi in phi:
                    content = content.replace(each_phi, phi_dic[each_phi])
            #else:    
                #print(content)
                #print('\n')
            
            deid_transcript.write(row['stimulus'])
            deid_transcript.write('\t')
            #print(transcript)
            #print(row['speaker'])
            deid_transcript.write(row['speaker'])
            deid_transcript.write('\t')
            deid_transcript.write(str(row['start_time']))
            deid_transcript.write('\t')
            deid_transcript.write(str(row['end_time']))
            deid_transcript.write('\t')
            deid_transcript.write(content)
            deid_transcript.write('\n')

