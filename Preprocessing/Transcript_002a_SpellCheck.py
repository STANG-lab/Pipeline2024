#!/usr/bin/env python
# coding: utf-8

# # spell check and flag the differences

# In[110]:


# note: https://theautomatic.net/2019/12/10/3-packages-to-build-a-spell-checker-in-python/
get_ipython().run_line_magic('pip', 'install pyspellchecker')


# In[1]:


import re
from spellchecker import SpellChecker
 
spell = SpellChecker()


# In[112]:


# find those words that may be misspelled
misspelled = spell.unknown(['let,', 'us.', 'wakl','on','the','ground'])
split = spell.split_words('let. us wlak.')
for word in misspelled:
    # Get the one `most likely` answer
    print(spell.correction(word))

    # Get a list of `likely` options
    print(spell.candidates(word))
    
print('split: ', split)
print('misspelled: ', misspelled)


# In[2]:


import pandas as pd
import numpy as np
import os


# In[7]:


### Points to the folder that contains all data
root_data = "C://Users//ANikzad//Desktop//Local_Pipeline//Data//"

### Specify Source Folder
    # --- Remora
std_transcript_location = "Remora-2023//Batch-1//1_std_transcripts//"

spell_check_output_path = root_data+"Remora-2023//Batch-1//reports//spell_check//"

drive_trancript_in_path = root_data+std_transcript_location


# In[115]:


import collections
from collections import defaultdict


# In[116]:


# get all the files in the transcripts folder (the folder should just include transcripts)
all_transcript_files = os.listdir(drive_trancript_in_path)
all_df = pd.DataFrame(columns=['filename', 'line'])
bug_dict = collections.defaultdict(list)
n = 0
for filepath in all_transcript_files:    
    df = pd.read_csv(drive_trancript_in_path + filepath, sep='\t', names=['none','speaker','start_t', 'end_t','content'])
    df['row_num'] = df.index
    for index, row in df.iterrows():
        #print(filepath)
        content = row['content']
        #print(content)
        partials = re.findall(r'[a-zA-Z]*\-\s', content)
        if partials != []:
            for each_item in partials:
              content = content.replace(each_item,'')
        partials = re.findall(r'[a-zA-Z]*\-[a-zA-Z]*', content)
        if partials != []:
            for each_item in partials:
              content = content.replace(each_item,'')
        partials = re.findall(r'[a-zA-Z]*\-', content)
        if partials != []:
            for each_item in partials:
              content = content.replace(each_item,'')
        neologisms = re.findall(r'[a-zA-Z]*\^', content)
        if neologisms != []:
            for each_item in neologisms:
              content = content.replace(each_item,'')
        #print(content)
        all_tokens = spell.split_words(content)
        #print(curr)
        non_partial_tokens = []
        df.at[index,'line'] = index
        df.at[index, 'filename'] = filepath.split('.')[0]
        
        for each_token in all_tokens:
           if each_token[-1] != '-':
              non_partial_tokens.append(each_token)

        misspelled = spell.unknown(non_partial_tokens)

        if len(misspelled) > 0:
          for word in misspelled:
            if len(word) > 1 :
              if word not in ['hm', 'nsv', "you're", 'xxx', 'bibliography', 'covetousness', 'tv', 'ajax', 'mhm', 'mm',
                              "patsy's", "toolbar", "covid", "mmm", "northwell", "microsoft", "pdf", ]:
                if spell.correction(word) != None:

                  df.at[index,'corrected'] = spell.correction(word)
                  df.at[index,'misspelled'] = word

                  n+=1

    all_df = pd.concat([all_df,df], ignore_index=True)
all_df


# In[44]:


n


# In[45]:


all_df.loc[~all_df['corrected'].isnull()]


# In[41]:


all_df.loc[~all_df['corrected'].isnull()].to_csv(spell_check_output_path+'spellcheck_df.csv')


# In[9]:


spell_check_df = pd.read_excel(spell_check_output_path+'spellcheck_REMORA_0823_LB_SB.xlsx')
spell_check_df = spell_check_df.loc[~spell_check_df['corrected'].isnull()].copy().drop('Unnamed: 0', axis=1)
correction_dic = {}
correction_list = []
for index, row in spell_check_df.iterrows():
    if row['filename'] not in correction_list:
        correction_list.append(row['filename'])
    locate_tupple = (row['filename'], row['start_t'])
    correction_tupple = (row['misspelled'], row['corrected'])
    correction_dic[locate_tupple] = correction_tupple


# In[10]:


### Points to the folder that contains all data
root_data = "C://Users//ANikzad//Desktop//Local_Pipeline//Data//"

### Specify Source Folder
    # --- Remora
std_transcript_location = "Remora-2023//Batch-1//1_std_transcripts//"

### Specify Destination folder
    # --- Remora
spellchecked_transcript_location = "Remora-2023//Batch-1//1b_std_transcripts_spellchecked//"


# In[11]:


all_transcript_files = os.listdir(root_data+std_transcript_location)
len(all_transcript_files)


# In[12]:


speaker_dic = {
    'Subject': 'Subject',
    'Patient': 'Subject',
    'Interviewer': 'Interviewer',
    'Unknown 1':'Other',
    'Unknown 2':'Other',
    'Computer':'Other',
    'Unknown':'Other',
    'Other': 'Other',
    'Unknown1': 'Other'

}

speakers_list = ['Subject', 'Interviewer']


# In[13]:


for filepath in all_transcript_files:#[0:1]:
    with open(root_data+spellchecked_transcript_location+filepath, 'w') as corrected_transcript:
        with open(drive_trancript_in_path+filepath, 'r') as source_file:
            if filepath.split('.')[0] in correction_list:
                for line in source_file:
                    each_line = line.strip().split('\t')
                    locate_tupple = (filepath.split('.')[0], float(each_line[2]))
                    content = each_line[4]
                    #print(locate_tupple)
                    if locate_tupple in correction_dic.keys():
                        print(locate_tupple)
                        print(content)
                        print(correction_dic[locate_tupple], correction_dic[locate_tupple][0], correction_dic[locate_tupple][1])
                        content = content.replace(correction_dic[locate_tupple][0], correction_dic[locate_tupple][1])
                        print(content)
                        print('\n')
                    corrected_transcript.write(each_line[0])
                    corrected_transcript.write('\t')
                    print(filepath, each_line[1])
                    corrected_transcript.write(speaker_dic[each_line[1]])
                    corrected_transcript.write('\t')
                    corrected_transcript.write(each_line[2])
                    corrected_transcript.write('\t')
                    corrected_transcript.write(each_line[3])
                    corrected_transcript.write('\t')
                    corrected_transcript.write(content)
                    corrected_transcript.write('\n')

            else:
                for line in source_file:
                    each_line = line.strip().split('\t')
                    corrected_transcript.write(each_line[0])
                    corrected_transcript.write('\t')
                    print(filepath, each_line[1])
                    corrected_transcript.write(speaker_dic[each_line[1]])
                    corrected_transcript.write('\t')
                    corrected_transcript.write(each_line[2])
                    corrected_transcript.write('\t')
                    corrected_transcript.write(each_line[3])
                    corrected_transcript.write('\t')
                    corrected_transcript.write(each_line[4])
                    corrected_transcript.write('\n')
        #if filepath.split('.')[0] in correction_list:


# In[ ]:





# Remove the subject
# remora_13095_T1_13095T101_v2023_BTW_none

# In[62]:


"Rebeccca's birthday is approaching. She says to her dad I love animals especially dogs.".replace("Rebeccca's", "Rebecca's")

