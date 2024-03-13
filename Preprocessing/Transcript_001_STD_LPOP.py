#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import re


# In[1]:


### Points to the folder that contains all data
root_data = "C://Users//ANikzad//Desktop//Local_Pipeline//Data//"

### Specify Input Source Folder
    ### --- Remora
source_location = "Remora-2023//Batch-1//0_source_transcripts//Transcripts-Reviewed-2021-06//"
    ### --- 

### Specify Output Folder
    # --- Remora
std_transcript_location = "Remora-2023//Batch-1//1_std_transcripts//"


# In[3]:


all_transcript_files = os.listdir(root_data+source_location)
len(all_transcript_files)


# REMOVE WRAT / ER40

# In[4]:


### Identify and correct transcripts that deviate from their generic source structure
for each_transcript in all_transcript_files:
    each_address = root_data+source_location+each_transcript
    each_transcript_df = pd.DataFrame()
    n=0
    with open(each_address,'r') as transcript:
        for each_line in transcript.readlines():
            n=n+1
            try:
                each_df = pd.DataFrame([each_line.strip().split('\t')], columns=['transcriber', 'none', 'start_time', 'end_time','task','speaker','content'])
            except:
                print(each_transcript)
                print('#line:', n)
                print(each_line)
                print(each_line.strip().split('\t'))
                print('\n', '###')

### All Errors fixed in Remora - see std-report folder


# In[5]:


task_dic ={'FFluency': 'FLU', 'Animals':'FLU', 'none':'BTW', 
           'TATPicture': 'PIC', 'CookieTheft': 'PIC', 'Rorschach8': 'PIC',
           'AboutYourself': 'JOU', 'HowsItGoing': 'JOU',
           'Paragraph': 'PAR', 'Hinting':'SOC'}

sample_dic ={'none': '01', 
             'FFluency': '11', 'Animals':'12', 
             'Paragraph': '21',
            'AboutYourself': '31', 'HowsItGoing': '32',
             'CookieTheft': '41','Rorschach8': '42', 'TATPicture': '43', 
             'Hinting':'51'}


timepoint_dic = {'BL':'T1','FU': 'T2'}


# Filename Structure:
# study_grid_timepoint_sampleid_version_task_stimulus

# In[6]:


nsv_dic = { '{cough}': '{NSV}',
 '{lipsmack}': '{NSV}',
 '{breath}': '{NSV}',
 '{laugh}': '{laugh}',
 '{NSV}': '{NSV}',
 '{clap}': '{NSV}',
 '{laughing}': '{laugh}',
 '{yawn}': '{NSV}',
 '{NSW}': '{NSV}',
 '{lispmack}': '{NSV}',
 '{foreign}': '{NSV}',
 '{whispers}': '{NSV}',
 '{sniffle}': '{NSV}',
 '{name}': '{NSV}',
 '{Laugh}' : '{laugh}',
 '{sigh}': '{NSV}' }


# In[9]:


### 
all_nsv = []
for each_transcript in all_transcript_files:
    each_address = root_data+source_location+each_transcript
    each_transcript_df = pd.DataFrame()
    n=0
    with open(each_address,'r') as transcript:
        for each_line in transcript.readlines():
            n=n+1
            each_line_df = pd.DataFrame([each_line.strip().split('\t')], columns=['transcriber', 'none', 'start_time', 'end_time','task','speaker','content'])
            each_transcript_df = pd.concat([each_transcript_df, each_line_df], ignore_index=True)
    for each_task in each_transcript_df['task'].unique():
        task_df = each_transcript_df.loc[each_transcript_df['task'] == each_task].reset_index()[['task', 'speaker', 'start_time', 'end_time', 'content']].copy()
        if each_task == '':
            stimulus = 'none'
        else:
            stimulus = each_task
        study = 'remora'
        grid = each_transcript.split('_')[0]
        time_point = timepoint_dic[each_transcript.split('_')[1]]
        sample_id = grid+time_point+sample_dic[stimulus]
        version = 'v2023'
        task = task_dic[stimulus]
        filename = study+'_'+grid+'_'+time_point+'_'+sample_id+'_'+version+'_'+task+'_'+stimulus

        with open(root_data+std_transcript_location+filename+'.txt', 'w') as std_transcript:
        ####    
            for index, row in task_df.iterrows():
                content = row['content']

                ### identify and correct error tags such as (() or ()) etc.
                tag_error = re.findall(r'-\(', content)
                for each_item in tag_error:
                    print(filename, each_item)
                
                ### replace empty unintelligible with xxx
                content = content.replace('(())','xxx')
                
                ### replace unintelligible with what is transcribed 
                uninteg = re.findall(r'\(\(.*?\)\)', content)
                if uninteg != []:
                    for each_item in uninteg:
                        #print(each_item, '$$$', each_item[2:-2])
                        #print(filename)
                        #print(content)
                        content = content.replace(each_item, each_item[2:-2])
                        #print(content)
                        #print('\n')
                        
                ### Check punctuation/repeition
                #rep = re.findall(r'\.\)', content)
                #if rep != []:
                 #   for each_item in rep:
                  #      print(filename)
                   #     print(content)
                    #    print('\n')

                
                ### convert repetition tags
                # Single token repetitions
                repetition = re.findall(r'\([a-zA-Z]*?\)\=', content)
                if repetition != []:
                    for each_item in repetition:
                        #print(filename)
                        #print(content)
                        content = content.replace(each_item, each_item.replace('(','').replace(')',''))
                        #print(content)
                        #print('\n')
                repetition = re.findall(r'\([a-zA-Z]*?\)\=', content)
                # Multiple token repetitions
                repetition = re.findall(r'\(.*?\)\=', content)
                if repetition != []:
                    for each_item in repetition:
                        #print(filename)
                        #print(content)
                        content = content.replace(each_item, each_item.replace('(','').replace(')','').replace(' ', '= ').replace('.','= .').replace(',','= ,').replace('?','= ?'))
                        #print(content)
                        #print('\n')
                
                ### Make sure there is no more paranthesis after cleaning unintelligible and repetitions
                repetition = re.findall(r'\(', content)
                if repetition != []:
                    for each_item in repetition:
                        print(filename)
                        print(content)

                ### convert {}s to {NSV} or {laugh}
                nsv = re.findall(r'\{.*?\}', content)
                if nsv != []:
                    for each_item in nsv:
                        #print(filename)
                        #print(content)
                        content = content.replace(each_item, nsv_dic[each_item])
                        #print(content)
                        #print('\n')          

                ### Check compound words
                #compound = re.findall(r'[a-zA-Z]\-[a-zA-Z]', content)
                #if compound != []:
                #   for each_item in compound:
                #      print(filename)
                #     print(content)
                    #    print('\n')          

                ### Check Acronyms
                acronym = re.findall(r'(\~[a-zA-Z]\s\~[a-zA-Z]\s\~[a-zA-Z]\s\~[a-zA-Z]\s\~[a-zA-Z]\s\~[a-zA-Z]\s\~[a-zA-Z])', content)
                if acronym != []:
                    for each_item in acronym:
                        #print(filename)
                        #print(content)
                        content = content.replace(each_item, each_item.replace(' ~', '').replace('~', ''))
                        #print(content)
                        #print('\n')          
                acronym = re.findall(r'(\~[a-zA-Z]\s\~[a-zA-Z]\s\~[a-zA-Z]\s\~[a-zA-Z]\s\~[a-zA-Z]\s\~[a-zA-Z])', content)
                if acronym != []:
                    for each_item in acronym:
                        #print(filename)
                        #print(content)
                        content = content.replace(each_item, each_item.replace(' ~', '').replace('~', ''))
                        #print(content)
                        #print('\n')
                acronym = re.findall(r'(\~[a-zA-Z]\s\~[a-zA-Z]\s\~[a-zA-Z]\s\~[a-zA-Z]\s\~[a-zA-Z])', content)
                if acronym != []:
                    for each_item in acronym:
                        #print(filename)
                        #print(content)
                        content = content.replace(each_item, each_item.replace(' ~', '').replace('~', ''))
                        #print(content)
                        #print('\n')
                acronym = re.findall(r'(\~[a-zA-Z]\s\~[a-zA-Z]\s\~[a-zA-Z]\s\~[a-zA-Z])', content)
                if acronym != []:
                    for each_item in acronym:
                        #print(filename)
                        #print(content)
                        content = content.replace(each_item, each_item.replace(' ~', '').replace('~', ''))
                        #print(content)
                        #print('\n')
                acronym = re.findall(r'(\~[a-zA-Z]\s\~[a-zA-Z]\s\~[a-zA-Z])', content)
                if acronym != []:
                    for each_item in acronym:
                        #print(filename)
                        #print(content)
                        content = content.replace(each_item, each_item.replace(' ~', '').replace('~', ''))
                        #print(content)
                        #print('\n')
                acronym = re.findall(r'(\~[a-zA-Z]\s\~[a-zA-Z])', content)
                if acronym != []:
                    for each_item in acronym:
                        #print(filename)
                        #print(content)
                        content = content.replace(each_item, each_item.replace(' ~', '').replace('~', ''))
                        #print(content)
                        #print('\n')
                acronym = re.findall(r'(\~[a-zA-Z])', content)
                if acronym != []:
                    for each_item in acronym:
                        #print(filename)
                        #print(content)
                        content = content.replace(each_item, each_item.replace(' ~', '').replace('~', ''))
                        #print(content)
                        #print('\n')
                plus_sign = re.findall(r'\+', content)
                if plus_sign != []:
                    #print(filename)
                    #print(content)
                    content = content.replace('+','')
                    #print(content)
                    #print('\n')
                        
                print(filename)
                print(content)
                print('\n')
                
                std_transcript.write(stimulus)
                std_transcript.write('\t')
                std_transcript.write(row['speaker'])
                std_transcript.write('\t')
                std_transcript.write(row['start_time'])
                std_transcript.write('\t')
                std_transcript.write(row['end_time'])
                std_transcript.write('\t')
                std_transcript.write(content)
                std_transcript.write('\n')
        


# Second Turn of STD
#     Remove ER-40/WRAT
#     Change tasks name to match Winterlight

# In[3]:


import pandas as pd
import os
### Points to the folder that contains all data
root_data = "C://Users//ANikzad//Desktop//Local_Pipeline//Data//"

### Specify Input Source Folder
    ### --- Remora
source_location = "Remora-2023//Batch-1//0_source_transcripts//Transcripts-Standardized-2023-08//"
    ### --- 

### Specify Output Folder
    # --- Remora
std_transcript_location = "Remora-2023//Batch-1//1_std_transcripts//"


# In[31]:


all_transcript_files = os.listdir(root_data+source_location)
len(all_transcript_files)


# In[25]:


task_dic ={'FFluency': 'FLU', 'Animals':'FLU', 'none':'BTW', 
           'TATPicture': 'PIC', 'CookieTheft': 'PIC', 'Rorschach8': 'PIC',
           'AboutYourself': 'JOU', 'HowsItGoing': 'JOU',
           'Paragraph': 'PAR', 'Hinting':'SOC'}

sample_dic ={'none': '01', 
             'FFluency': '11', 'Animals':'12', 
             'Paragraph': '21',
            'AboutYourself': '31', 'HowsItGoing': '32',
             'CookieTheft': '41','Rorschach8': '42', 'TATPicture': '43', 
             'Hinting':'51'}


# In[34]:


for each_transcript in all_transcript_files:
    transcript_info = each_transcript.split('.')[0].split('_')
    if transcript_info[6] not in ['WRAT', 'ER40']:
        #print(transcript_info)
        new_name = []
        new_name.append(transcript_info[0])
        new_name.append(transcript_info[1])
        new_name.append(transcript_info[2])
        new_name.append(transcript_info[1]+transcript_info[2]+sample_dic[transcript_info[6]])
        new_name.append(transcript_info[4])
        new_name.append(task_dic[transcript_info[6]])
        new_name.append(transcript_info[6])
        #print(new_name)
        #print('\n')
        each_address = root_data+source_location+each_transcript
        each_transcript_df = pd.DataFrame()
        with open(each_address,'r') as transcript:
            for each_line in transcript.readlines():
                each_line_df = pd.DataFrame([each_line.strip().split('\t')], columns=['stimulus', 'speaker', 'start_time', 'end_time','content'])
                each_transcript_df = pd.concat([each_transcript_df, each_line_df], ignore_index=True)

        with open(root_data+std_transcript_location+'_'.join(new_name)+'.txt', 'w') as std_transcript:
            for index, row in each_transcript_df.iterrows():
                content= row['content']
                print(content)
                content = content.replace("’", "'")
                content = content.replace("â€™", "'")
                print(content)
                std_transcript.write(row['stimulus'])
                std_transcript.write('\t')
                std_transcript.write(row['speaker'])
                std_transcript.write('\t')
                std_transcript.write(row['start_time'])
                std_transcript.write('\t')
                std_transcript.write(row['end_time'])
                std_transcript.write('\t')
                std_transcript.write(content)
                std_transcript.write('\n')

