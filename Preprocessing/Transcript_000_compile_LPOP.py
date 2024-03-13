#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import re
import json


# In[30]:


###LPOP

recording_path = 'C:\\Users\\ANikzad\\Desktop\\Local_Pipeline\\Data\\Winterlight_LPOP\\Batch1\\Recording\\'
json_path = 'C:\\Users\\ANikzad\\Desktop\\Local_Pipeline\\Data\\Winterlight_LPOP\\Batch1\\Transcript_JSON\\LPOP_transcripts_2023_05_15.csv'
revised_transcripts_path_v3 = 'C:\\Users\\ANikzad\\Desktop\\Local_Pipeline\\Data\\Winterlight_LPOP\\Batch1\\source_transcript\\v3\\'
revised_transcripts_path_v2 = 'C:\\Users\\ANikzad\\Desktop\\Local_Pipeline\\Data\\Winterlight_LPOP\\Batch1\\source_transcript\\v2\\'
revised_transcripts_path_v1 = 'C:\\Users\\ANikzad\\Desktop\\Local_Pipeline\\Data\\Winterlight_LPOP\\Batch1\\source_transcript\\v1\\'
all_audio_files = os.listdir(recording_path)
output_path =  'C:\\Users\\ANikzad\\Desktop\\Local_Pipeline\\Data\\Winterlight_LPOP\\Batch1\\std_transcripts\\'


# In[31]:


df = pd.read_csv(json_path)
df.tail(2)


# In[32]:


df_nomen = pd.DataFrame(columns=['filename','study','uid', 'timepoint','sample_id','project','category', 'task'])
for each_file in all_audio_files:
    entry = []
    each_file_name = each_file.split('.')[0]
    each_file_list = each_file_name.split('_')
    entry.append(each_file_name)
    for i in each_file_list:
        entry.append(i)
    #print(entry)
    entry = pd.DataFrame([entry], columns= ['filename','study','uid', 'timepoint','sample_id','project','category', 'task'])
    df_nomen = pd.concat([df_nomen, entry], ignore_index=True)
df_nomen


# In[33]:


nomen_key = {}
for index, row in df_nomen.iterrows():
    nomen_key[int(row['sample_id'])] = row['filename']


# In[34]:


missed_samples = []
for index, row in df.iterrows():
    try:
        filename = nomen_key[row['sample_id']]
        df.at[index, 'audio_filename'] = filename
    except:
        missed_samples.append(row['sample_id'])
df.head(2)


# In[35]:


missed_samples


# In[36]:


df['json_utterances']


# In[37]:


all_transcript_files_v3 = os.listdir(revised_transcripts_path_v3)
len(all_transcript_files_v3)


# In[38]:


all_transcript_files_v2 = os.listdir(revised_transcripts_path_v2)
len(all_transcript_files_v2)


# In[39]:


all_transcript_files_v1 = os.listdir(revised_transcripts_path_v1)
len(all_transcript_files_v1)


# In[40]:


compiled_data=pd.DataFrame()
added_transcripts = []
for each_transcript in all_transcript_files_v3:
    added_transcripts.append(each_transcript.split('_v')[0])
    file_path = revised_transcripts_path_v3+each_transcript
    with open(file_path) as f:
        for l in f.readlines():
            line_df = pd.DataFrame(data=[l.split('\t')], columns = ['uid', 'sample_id' , 'unknown', 'start_time','end_time', 'task','speaker','content'])
            line_df['version'] = 'v3'
            compiled_data = pd.concat([compiled_data, line_df], axis=0, ignore_index=True)

for each_transcript in all_transcript_files_v2:
    if each_transcript.split('_v')[0] not in added_transcripts:
        added_transcripts.append(each_transcript.split('_v')[0])
        file_path = revised_transcripts_path_v2+each_transcript
        with open(file_path) as f:
            for l in f.readlines()[1:]:
                line_df = pd.DataFrame(data=[l.split('\t')], columns = ['uid', 'sample_id' , 'unknown', 'start_time','end_time', 'task','speaker','content'])
                line_df['version'] = 'v2'
                compiled_data = pd.concat([compiled_data, line_df], axis=0, ignore_index=True)
for each_transcript in all_transcript_files_v1:
    if each_transcript.split('_v')[0] not in added_transcripts:
        added_transcripts.append(each_transcript.split('_v')[0])
        file_path = revised_transcripts_path_v1+each_transcript
        with open(file_path) as f:
            for l in f.readlines()[1:]:
                line_df = pd.DataFrame(data=[l.split('\t')], columns = ['uid', 'sample_id' , 'unknown', 'start_time','end_time', 'task','speaker','content'])
                line_df['version'] = 'v1'
                compiled_data = pd.concat([compiled_data, line_df], axis=0, ignore_index=True)


# In[41]:


len(added_transcripts)


# In[42]:


len(compiled_data['sample_id'].unique())


# In[1]:


recompiled_list = []
for each_sample in compiled_data['sample_id'].unique():
    sample_df = compiled_data.loc[compiled_data['sample_id'] == each_sample].copy().reset_index()
    version = sample_df['version'].unique()[0]
    sample_info = df.loc[df['sample_id'] == int(each_sample)].copy()
    duration = sample_info['diarised_audio_duration'].unique()[0]
    file_name = sample_info['audio_filename'].unique()[0]
    #print(file_name)
    file_name = file_name.replace('_WLL_', f'_{version}_')
    n = len(sample_df)
    for index, row in sample_df.iterrows():
        if index+1 == n:
            if row['end_time'] == '0.0':
                sample_df.at[index, 'end_time'] = duration
    all_lines = []
    for index,row in sample_df.iterrows():
        each_line = []
        each_line.append(row['uid'])
        each_line.append(row['sample_id'])
        each_line.append(row['task'])
        each_line.append(row['start_time'])
        each_line.append(row['end_time'])
        each_line.append(row['speaker'])
        content = row['content'].replace('\n','')
        content = content.replace('&','')
        

        ### Check Acronyms
        acronym = re.findall(r'(\~[a-zA-Z]\s\~[a-zA-Z]\s\~[a-zA-Z]\s\~[a-zA-Z])', content)
        if acronym != []:
            for each_item in acronym:
                #print(file_name)
                #print(content)
                content = content.replace(each_item, each_item.replace(' ~', '').replace('~', ''))
                #print(content)
                #print('\n')      
        acronym = re.findall(r'(\~[a-zA-Z]\s\~[a-zA-Z]\s\~[a-zA-Z])', content)
        if acronym != []:
            for each_item in acronym:
                #print(file_name)
                #print(content)
                content = content.replace(each_item, each_item.replace(' ~', '').replace('~', ''))
                #print(content)
                #print('\n') 
        acronym = re.findall(r'(\~[a-zA-Z]\s\~[a-zA-Z])', content)
        if acronym != []:
            for each_item in acronym:
                #print(file_name)
                #print(content)
                content = content.replace(each_item, each_item.replace(' ~', '').replace('~', ''))
                #print(content)
                #print('\n') 
        acronym = re.findall(r'(\~[a-zA-Z])', content)
        if acronym != []:
            for each_item in acronym:
                #print(file_name)
                #print(content)
                content = content.replace(each_item, each_item.replace(' ~', '').replace('~', ''))
                #print(content)
                #print('\n') 
        
        
        ### replace unintelligible with what is transcribed 
        uninteg = re.findall(r'\(\(.*?\)\)', content)
        if uninteg != []:
            for each_item in uninteg:
                #print(each_item, '$$$', each_item[2:-2])
                #print(file_name)
                #print(content)
                content = content.replace(each_item, each_item[2:-2])
                #print(content)
                #print('\n')
        
        ### convert repetition tags
        # Single token repetitions
        repetition = re.findall(r'\([a-zA-Z]*?\)\=', content)
        if repetition != []:
            for each_item in repetition:
                #print(file_name)
                #print(content)
                content = content.replace(each_item, each_item.replace('(','').replace(')',''))
                #print(content)
                #print('\n')
        repetition = re.findall(r'\([a-zA-Z]*?\)\=', content)
        #Multiple token repetitions
        repetition = re.findall(r'\(.*?\)\=', content)
        if repetition != []:
            for each_item in repetition:
                #print(file_name)
                #print(content)
                content = content.replace(each_item, each_item.replace('(','').replace(')','').replace(' ', '= ').replace('.','= .').replace(',','= ,').replace('?','= ?'))
                #print(content)
                #print('\n')

        #Space before/after restarts 
        repetition = re.findall(r'[^\s]\#', content)
        if repetition != []:
            for each_item in repetition:
                #print(file_name)
                #print(content)
                content = content.replace(each_item, each_item.replace('#',' #'))
                #print(content)
                #print('\n')
        repetition = re.findall(r'\#[^\s]', content)        
        if repetition != []:
            for each_item in repetition:
                #print(file_name)
                #print(content)
                content = content.replace(each_item, each_item.replace('#','# '))
                #print(content)
                #print('\n')

        content = content.replace('mm hm', 'mhm')
        
        content = content.replace("Ã¢Â€Â™", "'")
        
        repetition = re.findall(r'\=[a-zA-Z]', content)        
        if repetition != []:
            for each_item in repetition:
                print(file_name)
                print(content)
                #content = content.replace(each_item, each_item.replace('#','# '))
                #print(content)
                print('\n')
        
        
        #print(file_name, '\t', content)
        #print('\n')
        
        each_line.append(content)
        all_lines.append(each_line)
    output_address = output_path+file_name+'.txt'
    #with open(output_address, 'w') as f:
     #   for each_segment in all_lines:
      #      for each_item in each_segment:
       #         f.write("%s\t" %each_item)
        #    f.write("\n")
    
    
    recompiled_list.append(file_name)


# In[65]:


sample_df


# In[50]:


len(recompiled_list)


# In[52]:


decoded_data = []
for index, row in df.iterrows():
  subject_id = row['original_subject_id']
  sample_id = row['sample_id']
  task_name = row['stimulus_filename']
  json_ut = row['json_utterances']
  filename = row['audio_filename']
  duration = row['diarised_audio_duration']
  try:
    json_decoded = json.loads(json_ut)
    #print(json_decoded)
    for i in range (0, len(json_decoded)):
      for j in range (0, len(json_decoded[i]['tokens'])):
        per_time = []
        per_time.append(filename)
        per_time.append(subject_id)
        per_time.append(sample_id)
        per_time.append(task_name)
        per_time.append(json_decoded[i]['start_time'])
        per_time.append(duration)
        per_time.append(json_decoded[i]['tokens'][j]['type'])
        per_time.append(json_decoded[i]['tokens'][j]['value'])
        decoded_data.append(per_time)
  except:
    per_time = []
    per_time.append(filename)
    per_time.append(subject_id)
    per_time.append(sample_id)
    per_time.append(task_name)
    decoded_data.append(per_time)
    print(filename)
decoded_df = pd.DataFrame(data=decoded_data, columns=['filename','original_subject_id','sample_id','task_name','start_time','duration', 'type', 'token'])
decoded_df


# Note that there are rows with null transcriptions

# In[53]:


decoded_df.loc[decoded_df['sample_id'] == 109666]


# In[58]:


all_lines = []
for filename in decoded_df['filename'].unique():
    file_df = decoded_df.loc[decoded_df['filename'] == filename].copy()
    all_turns = file_df['start_time'].unique()
    n_turns = len(all_turns)
    print(filename)
    duration = file_df['duration'].unique()[0]

    for n_turn in range(0, n_turns):
        start_time = all_turns[n_turn]
        if n_turn != len(all_turns) - 1:
            end_time = all_turns[n_turn + 1]
        elif n_turn == len(all_turns) - 1:
            end_time = duration
        subject_id = file_df['original_subject_id'].unique()[0]
        task = file_df['task_name'].unique()[0]
        each_line = []
        each_utterence = []
        for index, row in file_df.loc[file_df['start_time'] == start_time].iterrows():
        #if row['type'] != 'unfilled_pause':
            each_utterence.append(row['token'])
        #print(each_utterence)
        each_line.append(subject_id)
        each_line.append(filename)
        each_line.append(start_time)
        each_line.append(end_time)
        each_line.append(task)
        each_line.append('Subject')
        each_line.append(' '.join(each_utterence))
        #print(each_line)
        all_lines.append(each_line)
        output_address = output_path+filename+'.txt'
    with open(output_address, 'w') as f:
        for each_segment in all_lines:
            for each_item in each_segment:
                f.write("%s\t" %each_item)
            f.write("\n")


# In[59]:


decoded_df


# Report

# In[17]:




