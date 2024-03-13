#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
import os
import string
import statistics
import numpy as np


# In[ ]:


from allennlp_models import pretrained
coref_labeler = pretrained.load_predictor("coref-spanbert")


# In[66]:


### Points to the folder that contains all data
root_data = "C://Users//ANikzad//Desktop//Local_Pipeline//Data//"

### Specify Source Folder
    # --- Remora
word_level_location = "Remora-2023//Batch-1//3_word_aggregates//"

### Specify Destination folder
    # --- Remora
dicourse_feature_location = "Remora-2023//Batch-1//6_discourse_level_features//"

drive_aggregate_in_path = root_data+word_level_location
drive_discourselevel_coref_out_path = root_data+dicourse_feature_location+"6a_coreferences//"

feature_folder = "Remora-2023//Batch-1//features//"


# In[3]:


# get all the files in the transcripts folder (the folder should just include transcripts)
all_aggregate_files = os.listdir(drive_aggregate_in_path)
len(all_aggregate_files)


# In[ ]:


### Coreference

speakers = {'Subject':'Participant', 'Interviewer': 'Interviewer', 'Other':'Other'}
for aggregate in all_aggregate_files:
    if aggregate.split('_')[5] in [ 'JOU', 'SOC', 'PIC', 'BTW']:
        #print(aggregate.split('_')[5])
        #print(drive_aggregate_in_path+aggregate)
        df = pd.read_csv(drive_aggregate_in_path+aggregate)
        df["row_id"] = df.index
        df = df.loc[df['is_unintelligable'] == 0].copy()
        df = df.loc[df['is_repetition'] == 0].copy()
        df = df.loc[df['is_partial'] == 0].copy()
        #df = df.loc[df['is_punctuation'] == 0].copy()
        df = df.loc[df['is_noise'] == 0].copy()
        df = df.loc[df['is_laugh'] == 0].copy()
        df = df.loc[df['is_filledpause'] == 0].copy()
        df = df[['uid', 'speaker','sentence_id', 'token_id', 'content',]].reset_index()
        #df = df.loc[df['speaker'] == 'Subject'].copy()
        all_turns = []
        turn_content = []
        for index, row in df.iterrows():
            if index == 0:
                current_speaker  = row['speaker']
                turn_content.append(row['content'])
            else:
                current_speaker = row['speaker']
                previous_speaker = df.at[index-1, 'speaker']
                if current_speaker == previous_speaker:
                    turn_content.append(row['content'])
                else: 
                    all_turns.append([speakers[previous_speaker], ' '.join(turn_content).replace('  ', ' ').replace(' .', '.').replace(' !', '!').replace(' ,', ',').replace(' ?', '?').replace('#', '').replace(" ' " , "'").replace(" '", "'").replace('  ', ' ')])
                    turn_content = []
                    turn_content.append(row['content'])
        all_turns.append([speakers[current_speaker], ' '.join(turn_content).replace('  ', ' ').replace(' .', '.').replace(' !', '!').replace(' ,', ',').replace(' ?', '?').replace('#', '').replace(" ' " , "'").replace(" '", "'").replace('  ', ' ')])
        turn_df = pd.DataFrame(all_turns, columns=['speaker', 'turn_content'])
        sample = []
        for index, row in turn_df.iterrows():
            sample.append(row['speaker']+': '+row['turn_content'])
            #print(row['speaker']+':', row['turn_content'])
        sample = '\n'.join(sample)        
        coref_pred = coref_labeler.predict(document=sample)
        clusters = coref_pred['clusters']
        tokens = coref_pred['document']
        n = 0
        token_index_coref = pd.DataFrame(columns=['token_index', 'token_coref', 'n_clusters','clusters'])
        for each_token in tokens:
            token_index_coref.at[n, 'token_index'] = n
            token_index_coref.at[n, 'token_coref'] = each_token
            n = n+1

        n = 0
        for each_cluster in clusters:
            #print(f'cluster_{n}')
            #print(each_cluster)
            for each_element in each_cluster:
                #print(each_element)
                for token_index in range(each_element[0], each_element[1]+1):
                    #print(token_index)
                    if token_index_coref.isnull().at[token_index, 'clusters']:
                        token_index_coref.at[token_index, 'clusters'] = f'cluster_{n}'
                    else:
                        token_index_coref.at[token_index, 'clusters'] = token_index_coref.at[token_index, 'clusters']+', '+f'cluster_{n}'
            n+=1
        for index, row in token_index_coref.iterrows():
            try:
                n_clusters = len(row['clusters'].split(','))
                token_index_coref.at[index,'n_clusters'] = n_clusters
            except:
                pass
        coref_df = pd.DataFrame()

        for index, row in token_index_coref.iterrows():
            if index+1 < len(token_index_coref):
                if token_index_coref.at[index+1, 'token_coref'] == ':':
                    current_speaker = row['token_coref']
                elif row['token_coref'] not in [':', '\n']:
                    entry = {}
                    entry['speaker'] = current_speaker
                    entry['content'] = row['token_coref']
                    entry['n_clusters'] = row['n_clusters']
                    entry['clusters'] = row['clusters']
                    coref_df = pd.concat([coref_df, pd.DataFrame([entry])], ignore_index=True)
            else:
                entry = {}
                entry['speaker'] = current_speaker
                entry['content'] = row['token_coref']
                entry['n_clusters'] = row['n_clusters']
                entry['clusters'] = row['clusters']
                coref_df = pd.concat([coref_df, pd.DataFrame([entry])], ignore_index=True)
        coref_df.to_csv(drive_discourselevel_coref_out_path+aggregate)


# In[67]:


# get all the files in the transcripts folder (the folder should just include transcripts)
all_aggregate_files = os.listdir(drive_discourselevel_coref_out_path)
len(all_aggregate_files)


# In[102]:


report_df = pd.DataFrame(columns=['filename', ])
for aggregate in all_aggregate_files:
    #print(aggregate)
    df = pd.read_csv(drive_discourselevel_coref_out_path+aggregate)
    filename = aggregate.split('.')[0]
    tokens_dic = {}
    clusters = []
    
    for index, row in df.iterrows():
        tokens_dic[index] = row['content']
        if row['n_clusters'] >= 1:
            for each_coref in row['clusters'].split(','):
                if each_coref not in clusters:
                    clusters.append(each_coref)
    
    c_n_utterance = 0

    noref_dist = 0
    noref_list = []

    for index, row in df.loc[df['speaker'] == 'Participant'].iterrows():
        if row['content'] in ['.', '!', '?']:
            c_n_utterance += 1
        if row['n_clusters'] >= 1:
            noref_list.append(noref_dist)
            noref_dist = 0
        else:
            noref_dist += 1
    
    cluster_number = len(clusters)
    #print(clusters)
    #print('all cluster number:', cluster_number)


    cluster_n = 0
    size_list = []
    max_instance = []
    max_distance = []

    for each_cluster in clusters:
        cluster_df = pd.DataFrame()
        
        last_mention = 0
        each_coref = []
        each_index = []
        max_list = []
        distance_list = []
        for index, row in df.iterrows():
            
            if type(row['clusters']) != float:
                cluster_list = row['clusters'].split(',')
                for each_item in cluster_list:
                    if each_item == each_cluster:
                        try:
                            if each_item in df.at[index+1, 'clusters'].split(','):
                                each_coref.append(row['content'])
                                each_index.append(index)
                            else:
                                cluster_df = pd.concat([cluster_df, pd.DataFrame([{'address':(each_index[0],each_index[-1]), 'speaker':row['speaker'], 'content': ' '.join(each_coref)}])], ignore_index=True)
                                each_coref = []
                                each_index = []
                        except:
                            each_coref.append(row['content'])
                            each_index.append(index)
                            cluster_df = pd.concat([cluster_df, pd.DataFrame([{'address':(each_index[0],each_index[-1]), 'speaker':row['speaker'], 'content': ' '.join(each_coref)}])], ignore_index=True)
                            each_coref = []
                            each_index = []
        
        filtered_cluster_df = cluster_df.iloc[1:,0:].loc[cluster_df['speaker'] == 'Participant']
        if len(filtered_cluster_df) > 0:
            cluster_n += 1
            #print(each_cluster)
            #print(cluster_df)
            size_list.append(len(cluster_df.loc[cluster_df['speaker'] == 'Participant']))
            for index, row in cluster_df.loc[cluster_df['speaker'] == 'Participant'].iterrows():
                max_list.append(row['address'][1] - row['address'][0] + 1)
            max_instance.append(max(max_list))
            for index, row in cluster_df.iterrows():
                if index == 0:
                    pass
                elif row['speaker'] == 'Participant':
                    distance_list.append(row['address'][0] - cluster_df.at[index-1, 'address'][1])
            max_distance.append(max(distance_list))


    entry = {}
    entry['filename'] = filename
    entry['c_n_utterance'] = c_n_utterance
    entry['c_n_clusters '] = cluster_n
    try:
        entry['c_s_clusters'] = cluster_n/c_n_utterance
    except:
        pass
    try:
        entry['c_md_size'] = statistics.median(size_list)
    except:
        pass
    try:
        entry['c_mx_size'] = np.percentile(size_list, 95)
    except:
        pass
    try:
        entry['c_md_instance'] = statistics.median(max_instance)
    except:
        pass
    try:
        entry['c_mx_instance'] = np.percentile(max_instance, 95)
    except:
        pass
    try:
        entry['c_mn_distance '] = statistics.mean(max_distance)
    except:
        pass
    try:
        entry['c_mx_distance'] = np.percentile(max_distance, 95)
    except:
        pass
    try:
        entry['c_mn_noref'] = statistics.mean(noref_list)
    except:
        pass
    try:
        entry['c_mx_noref'] = np.percentile(noref_list, 95)
    except:
        pass

    report_df = pd.concat([report_df, pd.DataFrame([entry])], ignore_index=True)

report_df.to_csv(root_data+feature_folder+"//6a_coref_features.csv")


# In[99]:


report_df


# In[94]:


df.head(10)


# In[35]:


df


# In[ ]:





# In[5]:


for aggregate in all_aggregate_files[0:2]:
    print(aggregate)
    df = pd.read_csv(drive_discourselevel_coref_out_path+aggregate)
    cluster_df = pd.DataFrame()
    tokens_dic = {}
    clusters = []
    for index, row in df.iterrows():
        tokens_dic[index] = row['content']
        if row['n_clusters'] >= 1:
            for each_coref in row['clusters'].split(','):
                if each_coref not in clusters:
                    clusters.append(each_coref)
    cluster_number = len(clusters)
    print('cluster number:', cluster_number)
    
    clusters_dic = {}
    for each_cluster in clusters:
        print(each_cluster)
        entry= {}
        entry['cluster'] = each_cluster
        #print(each_cluster)
        tokens = []
        for index, row in df.iterrows():
            if row['n_clusters'] >= 1:
                if each_cluster in row['clusters']:
                    tokens.append((index, row['content']))
        #print(tokens)
        merged_tokens = []
        token_numbers = []
        tokens.append(('null','null'))
        for j in range(0, len(tokens)-1):
            token_numbers.append(tokens[j][0])
            if tokens[j][0] + 1 != tokens[j+1][0]:
                merged_tokens.append((token_numbers[0], token_numbers[-1]))
                token_numbers = []
        print(merged_tokens)
        entry['size'] = len(merged_tokens)
        first_token =  merged_tokens[0][0]
        last_token =  merged_tokens[-1][1]
        print(first_token, last_token, last_token - first_token + 1)
        entry['max_distance'] = last_token - first_token + 1
        cluster_df = pd.concat([cluster_df, pd.DataFrame([entry])], ignore_index=1) 

    print('')


# In[9]:


cluster_df


# In[23]:


merged_tokens


# In[37]:


cluster_df


# In[18]:


clusters_dic


# In[11]:


clusters


# In[ ]:



 


# In[8]:


df


# In[115]:


### Embedding

speakers = {'Subject':'Participant', 'Interviewer': 'Interviewer', 'Other':'Other'}
for aggregate in all_aggregate_files [0:5]:
    if aggregate.split('_')[5] in [ 'JOU', 'SOC', 'PIC', 'BTW']:
        #print(aggregate.split('_')[5])
        #print(drive_aggregate_in_path+aggregate)
        df = pd.read_csv(drive_aggregate_in_path+aggregate)
        df["row_id"] = df.index
        df = df.loc[df['is_unintelligable'] == 0].copy()
        df = df.loc[df['is_repetition'] == 0].copy()
        df = df.loc[df['is_partial'] == 0].copy()
        #df = df.loc[df['is_punctuation'] == 0].copy()
        df = df.loc[df['is_noise'] == 0].copy()
        df = df.loc[df['is_laugh'] == 0].copy()
        df = df.loc[df['is_filledpause'] == 0].copy()
        df = df[['uid', 'speaker','sentence_id', 'token_id', 'content',]].reset_index()
        #df = df.loc[df['speaker'] == 'Subject'].copy()
        all_turns = []
        turn_content = []
        for index, row in df.iterrows():
            if index == 0:
                current_speaker  = row['speaker']
                turn_content.append(row['content'])
            else:
                current_speaker = row['speaker']
                previous_speaker = df.at[index-1, 'speaker']
                if current_speaker == previous_speaker:
                    turn_content.append(row['content'])
                else: 
                    all_turns.append([speakers[previous_speaker], ' '.join(turn_content).replace('  ', ' ').replace(' .', '.').replace(' !', '!').replace(' ,', ',').replace(' ?', '?').replace('#', '')])
                    turn_content = []
                    turn_content.append(row['content'])
        all_turns.append([speakers[current_speaker], ' '.join(turn_content).replace('  ', ' ').replace(' .', '.').replace(' !', '!').replace(' ,', ',').replace(' ?', '?').replace('#', '')])
        turn_df = pd.DataFrame(all_turns, columns=['speaker', 'turn_content'])


# In[116]:


turn_df

