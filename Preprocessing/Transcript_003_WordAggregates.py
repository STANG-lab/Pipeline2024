#!/usr/bin/env python
# coding: utf-8

# # 3 Converts Each Transcript to a word-level df
# 
# Input:   transcript text files
# 
# output:  csv word-level files

# In[32]:


# import libraries
import numpy as np
import pandas as pd
import re
import string
import os
import math


# In[34]:


### Points to the folder that contains all data
root_data = "C://Users//ANikzad//Desktop//Local_Pipeline//Data//"

### Specify Source Folder
    # --- Remora
deid_transcript_location = "Remora-2023//Batch-1//2_deid_transcripts//"

### Specify Destination folder
    # --- Remora
word_level_location = "Remora-2023//Batch-1//3_word_aggregates//"
feature_folder = "Remora-2023//Batch-1//features//"

drive_trancript_in_path = root_data+deid_transcript_location
drive_out_path = root_data + word_level_location


# In[35]:


# get all the files in the transcripts folder (the folder should just include transcripts)
all_transcript_files = os.listdir(drive_trancript_in_path)
len(all_transcript_files)


# In[36]:


# a to string method to pretty print
def __str__(self):

  s = "\"" + str(self.word) + "\" " + "[" + str(self.uid) + ":" + str(self.speaker)
  s = s + "]"
  s = s + "(n=" + str(self.n_words)
  s = s if self.is_partial == 0 else s + " partial"
  s = s if self.is_repetition == 0 else s + " rep."
  s = s if self.is_neologism == 0 else s + " neolog."
  s = s if self.is_unintelligable == 0 else s + " unintell."
  s = s if self.is_noise == 0 else s + " noise"
  s = s if self.is_punctuation == 0 else s + " punct."
  s = s if self.is_phi == 0 else s + " identi."
  s = s + ")"
  return s


# In[37]:


# this class stores all the attributes that a transcript token (i.e., a unit separated by spaces (most of the time a word) or punctuation) 
# this class does use 0 for false and 1 for true to make it easier later to count (i.e., by summing)

class TranscriptToken:
  # constructor with defaults
  def __init__(self, 
            word, n_words = 0, uid = "", speaker = "", 
            sentence_id = "", token_id = "", 
            is_partial = 0, is_repetition = 0,
            is_neologism = 0, is_unintelligable = 0, is_noise = 0, is_laugh = 0, is_punctuation = 0, is_phi = 0,
            is_uh = 0, is_er = 0, is_um = 0, is_filledpause = 0):
    
    self.uid = uid # the transcript the token belongs to
    self.speaker = speaker # the speaker, e.g., 'I' or 'S'
    self.sentence_id = sentence_id # a unique id of the sentence
    self.token_id = token_id # a unique id for this token
    self.word = word # the transcript token
    self.n_words = n_words # number of words
    self.is_partial = is_partial # a partial word
    self.is_repetition = is_repetition # a repetition
    self.is_neologism = is_neologism # neologism
    self.is_unintelligable = is_unintelligable # unintelligable word or region
    self.is_noise = is_noise # non speach verbalisation
    self.is_laugh = is_laugh # laugh
    self.is_punctuation = is_punctuation # punctuation
    self.is_phi = is_phi # relevant for remora
    self.is_uh = is_uh
    self.is_er = is_er
    self.is_um = is_um
    self.is_filledpause = is_filledpause
  

  # a list of columns for the final dataframe
  column_list = ["uid" ,
              "speaker",
              "sentence_id" ,
              "token_id",
              "content" ,
              "n_words" ,
              "is_speech_pause",
              "is_partial" ,
              "is_repetition" ,
              "is_neologism",
              "is_speech_error",
              "is_unintelligable",
              "is_noise",
              "is_punctuation",
              "is_phi"]

  # turn the object into a dictionary
  def to_dictionary(self):
    token_dict = {
              "uid" : self.uid,
              "speaker" : self.speaker,
              "sentence_id" : self.sentence_id,
              "token_id" : self.token_id,
              "content" : self.word, # spreadsheet K colum changed from 'word' to 'content'; because it is not always words… I thought content is more appropriate
              "n_words" : self.n_words,
              "is_partial" : self.is_partial, 
              "is_repetition" : self.is_repetition, 
              "is_neologism": self.is_neologism,
              "is_unintelligible": self.is_unintelligable, 
              "is_noise": self.is_noise, 
              "is_laugh": self.is_laugh,
              "is_punctuation": self.is_punctuation,
              "is_phi": self.is_phi,
              "is_uh": self.is_uh,
              "is_er": self.is_er,
              "is_um": self.is_um,
              "is_filledpause": self.is_filledpause
          }
    return token_dict
# test
print(TranscriptToken(word = "abc", n_words=1).to_dictionary())


# In[41]:


#%pip install ply
import ply.lex as lex
import numpy as np

__file__ = "remora02_2_aggregate-v2.ipynb"

class TranscriptLexer(object):

  unintelligable = 0
  repetition = 0

  # unique id for each token (is incremented)
  __token_id = 0

  # quick method to get the current token id and at the same time increment it
  def token_id(self):
    temp = self.__token_id
    self.__token_id = self.__token_id + 1
    return temp

  # List of token names. This is always required. Each token name needs to have a method below with "t_" at the beginning. 
  # The order of the methods (not the items in this list) is important.
  # it is like an if else branch. 
  # The first methods it finds where the regex pattern fits, it will use to tag the token.
  tokens = (
    'REPETITION_SINGLE',
    'REPETITION_SINGLE_APO',
    "PHI",
    'NEOLOGISM',
    'NOISE',
    'LAUGH',
    'PARTIAL',
    'LETTER',
    'LETTER_PARTIAL',
    'UNINTELLIGABLE_WORD',
    'UNINTELLIGABLE_REGION_EMPTY',
    'SPEECH_ERROR',
    'PUNCTUATION',
    'UM',
    'UH',
    'ER',
    'WORD',
    'WORD_APO',
  )

  # an empty unintelligable region, i.e., (())
  def t_UNINTELLIGABLE_REGION_EMPTY(self,t):
    r'xxx'
    t.value = TranscriptToken(word = "xxx", token_id = self.token_id(), n_words=np.nan, is_unintelligable=1)
    return t
 
  # a single repetition word, e.g., "I I="
  def t_REPETITION_SINGLE(self,t): 
    r'[A-Za-z\.\,\^\?]+='
    t.value = TranscriptToken(word = t.value.strip()[:-1],token_id = self.token_id(), n_words=1, is_repetition=1, is_unintelligable=self.unintelligable)
    return t

  # repetitions with '
  def t_REPETITION_SINGLE_APO(self,t): 
    r'[A-Za-z-\']+='
    word = t.value.lower()

    t.value = TranscriptToken(word = t.value.strip()[:-1],token_id = self.token_id(), n_words=1, is_repetition=1, 
                                is_unintelligable=self.unintelligable)
    return t


  # matches any punctuation marks, i.e., ". , ! ? #"
  def t_PUNCTUATION(self,t):
    r'\.|\?|!|,|\#'
    t.value = TranscriptToken(word = t.value.strip(), token_id = self.token_id(), n_words=0, is_punctuation=1, 
                              is_unintelligable=self.unintelligable, is_repetition=self.repetition)
    return t

  

  # a token that contains PHI, e.g., "[Harry Potter]"
  def t_PHI(self,t):
    r'\[[A-Za-z\s\,\.\']+\]' #yan commented out 0914; 
    #r'\[[((A-Za-z.\s))]+\]' # to cover '[St. Louis]', '[((Rielle Nikia))]', '[Long Island]', '[India]', '[((Rielle. Nikia))]'
    #r'\[[A-Za-z.\s]+\]' 
    t.value = TranscriptToken(word = t.value.strip()[1:-1],token_id = self.token_id(), n_words=len(t.value.split(" ")), is_phi=1,
                              is_repetition = self.repetition, is_unintelligable=self.unintelligable)
    return t

  # neologism, e.g., "paperskate^"
  def t_NEOLOGISM(self,t):
    r'[A-Za-z]+\^'
    t.value = TranscriptToken(word = t.value.strip()[:-1],token_id = self.token_id(), n_words=1, is_neologism=1, 
                              is_repetition = self.repetition, is_unintelligable=self.unintelligable)
    return t

  # partial letters
  def t_LETTER_PARTIAL(self,t):
    r'~[A-Za-z~]+-\s'
    t.value = TranscriptToken(word = t.value.strip(),token_id = self.token_id(), n_words=1, is_single_letter=1, 
                              is_repetition = self.repetition, is_unintelligable=self.unintelligable)
    return t


  # noise, e.g., "{laugh}"
  def t_NOISE(self,t):
    r'{NSV}'
    t.value = TranscriptToken(word = t.value.strip(),token_id = self.token_id(), n_words=0, is_noise=1, 
                              is_repetition = self.repetition, is_unintelligable=self.unintelligable)
    return t

  def t_LAUGH(self,t):
    r'{laugh}'
    t.value = TranscriptToken(word = t.value.strip(),token_id = self.token_id(), n_words=0, is_laugh=1, 
                              is_repetition = self.repetition, is_unintelligable=self.unintelligable)
    return t

  def t_UM(self,t):
    r'um(?![a-zA-Z])'
    t.value = TranscriptToken(word = t.value.strip(),token_id = self.token_id(), n_words=0,
                              is_repetition = self.repetition, is_unintelligable=self.unintelligable, is_um =1, is_filledpause=1)
    return t
  
  def t_UH(self,t):
    r'uh(?![a-zA-Z])'
    t.value = TranscriptToken(word = t.value.strip(),token_id = self.token_id(), n_words=0, 
                              is_repetition = self.repetition, is_unintelligable=self.unintelligable, is_uh =1, is_filledpause=1)
    return t
  
  def t_ER(self,t):
    r'er(?![a-zA-Z])'
    t.value = TranscriptToken(word = t.value.strip(),token_id = self.token_id(), n_words=0, 
                              is_repetition = self.repetition, is_unintelligable=self.unintelligable, is_er =1, is_filledpause=1)
    return t


  # partial words
  def t_PARTIAL(self,t):
    r'[A-Za-z]+-(\s|$)'
    # r'[A-Za-z]+-\s'

    t.value = TranscriptToken(word = t.value.strip()[:-1],token_id = self.token_id(), n_words=1, is_partial=1, 
                              is_repetition = self.repetition, is_unintelligable=self.unintelligable)
    return t

  # words with ' #and \’ yan added
  def t_WORD_APO(self,t):
    r'[A-Za-z-\'\’]+' 
    word = t.value.lower()

    t.value = TranscriptToken(word = t.value.strip(),token_id = self.token_id(), n_words=1, is_unintelligable=self.unintelligable,
                              is_repetition = self.repetition)
    return t

  # words
  def t_WORD(self,t):
    r'[A-Za-z-]+'

    t.value = TranscriptToken(word = t.value.strip(),token_id = self.token_id(), n_words=1, is_unintelligable=self.unintelligable,
                              is_repetition = self.repetition)
    return t

  

  # Define a rule so we can track line numbers
  def t_newline(self,t):
      r'\n+'
      t.lexer.lineno += len(t.value)

  # A string containing ignored characters (spaces and tabs)
  t_ignore  = ' \t\n'

  # Error handling rule
  def t_error(self,t):
      print("Illegal character '%s'" % t.value[0])
      t.lexer.skip(1)

  # Build the lexer
  def reset(self,**kwargs):
      self.lexer = lex.lex(module=self, **kwargs)
      self.unintelligable = 0

  # Test it output
  def test(self,data):
      self.lexer.input(data)
      while True:
            tok = self.lexer.token()
            if not tok:
                break
            print(tok.value)
  
  def get_token_lexer(self,data):
    self.lexer.input(data)
    return self.lexer


# In[48]:


class ReadOnlyStack:

  _stack = None
  _top_idx = -1
  _idx_upper_bound = -1

  def __init__(self, data):
    self._stack = data
    self._top_idx = 0
    self._idx_upper_bound = len(data)

  def pop(self):
    if self._top_idx < self._idx_upper_bound:
      item = self._stack[self._top_idx]
      self._top_idx = self._top_idx + 1
      return item
    else:
      return None
  
  def peek(self):
    # print(self._top_idx)
    if self._top_idx < self._idx_upper_bound:
      item = self._stack[self._top_idx]
      return item
    else:
      return None

  def see_data(self):
    return self._stack

  def is_empty(self):
    return self._top_idx >= self._idx_upper_bound

def transform_t_df_token_stack(t_df, uid):
  token_list = list()
  m = TranscriptLexer()

  for i,row in t_df.iterrows():
    speaker = row["speaker"]
    content = row["content"]

    m.reset()           # Build the lexer
    token_lex = m.get_token_lexer(content)
    while True:
      tok = token_lex.token()

      if not tok:
          break

      token = tok.value
      token.uid = uid
      token.speaker = speaker
      token_list.append(token)

  token_stack = ReadOnlyStack(token_list)    
  return token_stack


# In[104]:


count = 0
debug = []
report_df = pd.DataFrame(columns=['filename', 'w_n_words', 'w_s_words','w_n_utterance', 
                                'w_n_unintelligible','w_w_unintelligible', 'w_n_repetition', 'w_w_repetition',
                                'w_n_partial', 'w_w_partial', 'w_n_comma','w_s_comma','w_n_period','w_s_period', 
                                'w_n_exclamation', 'w_s_exclamation','w_n_question','w_s_question',
                                'w_n_phi', 'w_w_phi', 'w_n_neologism','w_w_neologism', 'w_n_noise','w_w_noise',
                                'w_n_laugh','w_w_laugh', 'w_n_filledpause', 'w_w_filledpause', 'w_n_uh','w_w_uh','w_n_um','w_w_um', 'w_n_er','w_w_er',
                                'w_n_restart','w_s_restart', 'w_n_dysfluent', 'w_w_dysfluent'])
for transcript in all_transcript_files:

    final_token_df = pd.DataFrame(columns=['uid', 'speaker','sentence_id', 'token_id', 'content', 'n_words', 
                                           'is_unintelligable','is_repetition','is_partial','is_punctuation',
                                           'is_phi','is_neologism', 'is_noise', 'is_laugh', 'is_filledpause',
                                            'is_uh', 'is_um', 'is_er'])
    count += 1 
    in_cols = ['none', 'speaker', 'start_s', 'end_s', 'content']
    
    filepath = drive_trancript_in_path + transcript
    print(filepath)
    t_df = pd.read_csv(filepath, sep = "\t", names=in_cols)
    t_token_stack = transform_t_df_token_stack(t_df, transcript.split('.')[0])
    sentence_id = 0

    while not t_token_stack.is_empty():
        t_token = t_token_stack.pop()
        t_token.sentence_id = sentence_id
        final_token_df = pd.concat([final_token_df, pd.DataFrame.from_records(t_token.to_dictionary(), index=[0])], ignore_index=True)
        if t_token.word in [".", "?", "!"]:
            sentence_id = sentence_id + 1
    final_token_df.rename(columns={"is_unintelligable": "is_unintelligible"}, inplace=True)
    final_token_df.to_csv(drive_out_path+transcript.split('.')[0]+'.csv')
    

    w_n_words = final_token_df.loc[final_token_df['speaker'] == 'Subject']['n_words'].sum()
    w_n_utterance = len(final_token_df.loc[final_token_df['speaker'] == 'Subject']['sentence_id'].unique())
    try:
        w_s_words = w_n_words / w_n_utterance
    except:
        w_s_words = np.nan
    w_n_unintelligible = final_token_df.loc[final_token_df['speaker'] == 'Subject']['is_unintelligible'].sum()
    try:
        w_w_unintelligible = (w_n_unintelligible/w_n_words)*100
    except:
        w_w_unintelligible = np.nan
    w_n_repetition = final_token_df.loc[final_token_df['speaker'] == 'Subject']['is_repetition'].sum()
    try:
        w_w_repetition = (w_n_repetition/w_n_words)*100
    except:
        w_w_repetition= np.nan
    
    w_n_partial = final_token_df.loc[final_token_df['speaker'] == 'Subject']['is_partial'].sum()
    try:
        w_w_partial = (w_n_partial/w_n_words)*100
    except:
        w_w_partial= np.nan
    w_n_comma = len(final_token_df.loc[final_token_df['speaker'] == 'Subject'].loc[final_token_df['content'] == ','])
    try:
        w_s_comma = w_n_comma/w_n_utterance
    except:
        w_s_comma= np.nan
    
    w_n_period = len(final_token_df.loc[final_token_df['speaker'] == 'Subject'].loc[final_token_df['content'] == '.'])
    try:
        w_s_period = w_n_period/w_n_utterance
    except:
        w_s_period= np.nan
    w_n_exclamation = len(final_token_df.loc[final_token_df['speaker'] == 'Subject'].loc[final_token_df['content'] == '!'])
    try:
        w_s_exclamation = w_n_exclamation/w_n_utterance
    except:
        w_s_exclamation = np.nan   
    w_n_question = len(final_token_df.loc[final_token_df['speaker'] == 'Subject'].loc[final_token_df['content'] == '?'])
    try:
        w_s_question = w_n_question/w_n_utterance
    except:
        w_s_question= np.nan
    w_n_phi = final_token_df.loc[final_token_df['speaker'] == 'Subject']['is_phi'].sum()
    try:
        w_w_phi = (w_n_phi/w_n_words)*100
    except:
        w_w_phi = np.nan
    w_n_neologism = final_token_df.loc[final_token_df['speaker'] == 'Subject']['is_neologism'].sum()
    try:
        w_w_neologism = (w_n_neologism/w_n_words)*100
    except:
        w_w_neologism = np.nan
    w_n_noise = final_token_df.loc[final_token_df['speaker'] == 'Subject']['is_noise'].sum()
    try:
        w_w_noise = (w_n_noise/w_n_words)*100
    except:
        w_w_noise = np.nan
    w_n_laugh = final_token_df.loc[final_token_df['speaker'] == 'Subject']['is_laugh'].sum()
    try:
        w_w_laugh = (w_n_laugh/w_n_words)*100
    except:
        w_w_laugh = np.nan
    w_n_filledpause = final_token_df.loc[final_token_df['speaker'] == 'Subject']['is_filledpause'].sum()
    try:
        w_w_filledpause = (w_n_filledpause/w_n_words)*100
    except:
        w_w_filledpause = np.nan
    w_n_uh = final_token_df.loc[final_token_df['speaker'] == 'Subject']['is_uh'].sum()
    try:
        w_w_uh = (w_n_uh/w_n_words) *100
    except:
        w_w_uh = np.nan
    w_n_um = final_token_df.loc[final_token_df['speaker'] == 'Subject']['is_um'].sum()
    try:
        w_w_um = (w_n_um/w_n_words) * 100
    except:
        w_w_um = np.nan
    w_n_er = final_token_df.loc[final_token_df['speaker'] == 'Subject']['is_er'].sum()
    try:
        w_w_er = (w_n_er/w_n_words) *100
    except:
        w_w_er= np.nan
    w_n_restart = len(final_token_df.loc[final_token_df['speaker'] == 'Subject'].loc[final_token_df['content'] == '#'])
    try:
        w_s_restart = w_n_restart/w_n_utterance
    except:
        w_s_restart = np.nan
    w_n_dysfluent = w_n_repetition + w_n_partial + w_n_filledpause + w_n_restart
    try:
        w_w_dysfluent = (w_n_dysfluent/w_n_words) * 100
    except:
        w_w_dysfluent = np.nan
    each_report = pd.DataFrame([{'filename':transcript.split('.')[0], 'w_n_words':w_n_words,'w_s_words': w_s_words, 'w_n_utterance': w_n_utterance,
                                'w_n_unintelligible': w_n_unintelligible, 'w_w_unintelligible':w_w_unintelligible, 'w_n_repetition': w_n_repetition,'w_w_repetition':w_w_repetition ,
                                'w_n_partial': w_n_partial, 'w_w_partial':w_w_partial, 
                                'w_n_comma':w_n_comma,'w_s_comma':w_s_comma,'w_n_period':w_n_period,'w_s_period':w_s_period,
                                'w_n_exclamation':w_n_exclamation,'w_s_exclamation':w_s_exclamation,'w_n_question':w_n_question,'w_s_question':w_s_question, 'w_n_phi': w_n_phi,'w_w_phi':w_w_phi,
                                'w_n_neologism': w_n_neologism,'w_w_neologism':w_w_neologism, 'w_n_noise':w_n_noise, 'w_w_noise':w_w_noise,
                                'w_n_laugh':w_w_laugh,'w_w_laugh':w_w_laugh, 'w_n_filledpause': w_n_filledpause, 'w_w_filledpause':w_w_filledpause,
                                  'w_n_uh': w_n_uh,'w_w_uh':w_w_uh,'w_n_um':w_n_um,'w_w_um': w_w_um, 'w_n_er':w_n_er, 'w_w_er': w_w_er,
                                  'w_n_restart':w_n_restart,'w_s_restart':w_s_restart, 'w_n_dysfluent':w_n_dysfluent,'w_w_dysfluent':w_w_dysfluent }])
    report_df = pd.concat([report_df,each_report], ignore_index=True)
    #print(drive_out_path+transcript.split('.')[0]+'.csv')
report_df.to_csv(root_data+feature_folder+"//3_wordaggregate_features.csv")
 

