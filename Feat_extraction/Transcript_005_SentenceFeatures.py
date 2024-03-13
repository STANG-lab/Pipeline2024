#!/usr/bin/env python
# coding: utf-8


# In[95]:


import pandas as pd
import numpy as np
import os
import string
from dialog_tag import DialogTag
import ast
from allennlp_models import pretrained
import statistics


# In[ ]:


from spacy.lang.en import English

nlp = English()
import spacy

nlp = spacy.load("en_core_web_sm")

speechact_model = DialogTag("distilbert-base-uncased")
srl_labeler = pretrained.load_predictor("structured-prediction-srl-bert")
dep_parser = pretrained.load_predictor("structured-prediction-biaffine-parser")


# function string nan
def isNaN(string):
    return string != string


# In[3]:


### Points to the folder that contains all data
root_data = "C://Users//ANikzad//Desktop//Local_Pipeline//Data//"

### Specify Source Folder
# --- Remora
word_level_location = "Remora-2023//Batch-1//3_word_aggregates//"

### Specify Destination folder
# --- Remora
sentence_feature_location = "Remora-2023//Batch-1//5_sentence_level_features//"

connective_raw_location = (
    "Remora-2023//Batch-1//6_discourse_level_features//6d_connectives//raw_samples//"
)


feature_folder = "Remora-2023//Batch-1//features//"


drive_aggregate_in_path = root_data + word_level_location
drive_sentencelevel_out_path = root_data + sentence_feature_location
drive_wordlevel_out_path_connective = root_data + connective_raw_location


# In[5]:


# get all the files in the transcripts folder (the folder should just include transcripts)
all_aggregate_files = os.listdir(drive_aggregate_in_path)
len(all_aggregate_files)


# In[ ]:


for aggregate in all_aggregate_files:
    if aggregate.split("_")[5] in ["SOC", "PIC", "JOU", "BTW"]:
        # print(aggregate.split('_')[5])
        # print(drive_aggregate_in_path+aggregate)
        df = pd.read_csv(drive_aggregate_in_path + aggregate)
        df["row_id"] = df.index
        df = df.loc[df["is_unintelligable"] == 0].copy()
        df = df.loc[df["is_repetition"] == 0].copy()
        df = df.loc[df["is_partial"] == 0].copy()
        # df = df.loc[df['is_punctuation'] == 0].copy()
        df = df.loc[df["is_noise"] == 0].copy()
        df = df.loc[df["is_laugh"] == 0].copy()
        df = df.loc[df["is_filledpause"] == 0].copy()
        df = df[
            [
                "uid",
                "speaker",
                "sentence_id",
                "token_id",
                "content",
            ]
        ]

        sent_df = pd.DataFrame(
            columns=[
                "uid",
                "sentence_id",
                "token_id",
                "content",
            ]
        )
        sentence_id = 0
        token_id = 0

        for n, l in df.groupby(["uid", "sentence_id"]):
            # print(l) # grouped df of this sentence
            sent = l["content"].str.cat(
                sep=" "
            )  # pasting together the original sentence
            sent = (
                sent.replace("  ", " ")
                .replace(" .", ".")
                .replace(" !", "!")
                .replace(" ,", ",")
                .replace(" ?", "?")
                .replace(" ' ", "'")
                .replace(" '", "'")
                .replace("#", "")
                .replace("  ", " ")
            )
            # print(sent)
            dualog_tag = speechact_model.predict_tag(sent)
            # do the POS tagging
            speaker = l["speaker"].unique()[0]
            sent_parse = dep_parser.predict(sent)
            sent_srl = srl_labeler.predict(sentence=sent)
            assert len(sent_parse["words"]) == len(sent_srl["words"])

            for i in range(0, len(sent_parse["words"])):
                srl_tag = []
                for j in range(0, len(sent_srl["verbs"])):
                    srl_tag.append(sent_srl["verbs"][j]["tags"][i])
                content = sent_parse["words"][i]
                dep_pos = sent_parse["pos"][i]
                dep_dependancies = sent_parse["predicted_dependencies"][i]
                srl_tag = srl_tag

                element_df = pd.DataFrame()
                element_df.at[0, "uid"] = aggregate.split(".")[0]
                element_df.at[0, "speaker"] = speaker
                element_df.at[0, "sentence_id"] = sentence_id
                element_df.at[0, "token_id"] = token_id
                element_df.at[0, "content"] = content
                element_df.at[0, "dep.pos"] = dep_pos
                element_df.at[0, "dep.dependancy"] = dep_dependancies
                element_df.at[0, "dualog_tag"] = dualog_tag
                element_df.at[0, "srl_list"] = str(srl_tag)

                sent_df = pd.concat([sent_df, element_df], ignore_index=True)

                token_id += 1

            sentence_id += 1
        sent_df.to_csv(root_data + sentence_feature_location + aggregate)
        current_speaker = sent_df.at[0, "speaker"]
        each_turn = []
        print(aggregate)
        with open(
            drive_wordlevel_out_path_connective + aggregate.split(".")[0] + ".txt", "w"
        ) as sample_text:
            for n, l in sent_df.groupby(["sentence_id", "speaker"]):
                for index, row in l.iterrows():
                    if l["speaker"].unique()[0] == current_speaker:
                        each_turn.append(row["content"])
                    else:
                        sample_text.write(
                            " ".join(each_turn)
                            .replace("  ", " ")
                            .replace(" .", ".")
                            .replace(" !", "!")
                            .replace(" ,", ",")
                            .replace(" ?", "?")
                            .replace(" - ", "-")
                            .replace(" ' ", "'")
                            .replace(" '", "'")
                            .replace("#", "")
                            .replace("  ", " ")
                        )
                        sample_text.write("\n")
                        current_speaker = l["speaker"].unique()[0]
                        each_turn = []
                        each_turn.append(row["content"])
            sample_text.write(
                " ".join(each_turn)
                .replace("  ", " ")
                .replace(" .", ".")
                .replace(" !", "!")
                .replace(" ,", ",")
                .replace(" ?", "?")
                .replace(" - ", "-")
                .replace(" ' ", "'")
                .replace(" '", "'")
                .replace("#", "")
                .replace("  ", " ")
            )


# In[4]:


# get all the files in the transcripts folder (the folder should just include transcripts)
all_aggregate_files = os.listdir(drive_sentencelevel_out_path)
len(all_aggregate_files)


# In[117]:


dependancy_tags = []
dialog_tags = []
semantic_tags = []
report_df = pd.DataFrame(
    columns=[
        "filename",
    ]
)
for aggregate in all_aggregate_files:
    df = pd.read_csv(drive_sentencelevel_out_path + aggregate)
    df = df.loc[df["speaker"] == "Subject"].copy()
    df = df.loc[df["dep.pos"] != "PUNCT"].copy()
    for index, row in df.iterrows():
        if row["dep.dependancy"] not in dependancy_tags:
            dependancy_tags.append(row["dep.dependancy"].replace(" ", "").lower())
        if row["dualog_tag"] not in dialog_tags:
            dialog_tags.append(
                row["dualog_tag"]
                .replace(" ", "")
                .replace("/", "")
                .replace("(", "")
                .replace(")", "")
                .replace("-", "")
                .replace(",", "")
                .lower()
            )
        srl_list = ast.literal_eval(row["srl_list"].replace("I-", "").replace("B-", ""))
        if len(srl_list) > 0:
            for each_srl in srl_list:
                if each_srl != "O" and each_srl not in semantic_tags:
                    # print(each_srl.replace('I-','').replace('B-',''))
                    semantic_tags.append(each_srl)

for aggregate in all_aggregate_files:
    df = pd.read_csv(drive_sentencelevel_out_path + aggregate)
    filename = aggregate.split(".")[0]
    df = df.loc[df["speaker"] == "Subject"].copy()
    df = df.loc[df["dep.pos"] != "PUNCT"].copy()

    dependancy_dic = {}
    dependancy_dic_w = {}
    dialog_dic = {}
    dialog_dic_s = {}
    semantic_dic = {}
    n_totalunits = []
    n_totalelements = []

    for tag in dependancy_tags:
        dependancy_dic[tag.replace(" ", "").lower()] = 0
    for tag in dialog_tags:
        dialog_dic[
            tag.replace(" ", "")
            .replace("/", "")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "")
            .replace(",", "")
            .lower()
        ] = 0
    for tag in semantic_tags:
        semantic_dic[tag] = 0
    u_n_utterance = len(df["sentence_id"].unique())
    u_n_words = len(df)
    for each_sentence in df["sentence_id"].unique():
        sentence_df = df.loc[df["sentence_id"] == each_sentence].copy().reset_index()
        dialog_tag = (
            sentence_df["dualog_tag"]
            .unique()[0]
            .replace(" ", "")
            .replace("/", "")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "")
            .replace(",", "")
            .lower()
        )
        dialog_dic[dialog_tag] = dialog_dic[dialog_tag] + 1
        srl_len = len(ast.literal_eval(sentence_df.at[0, "srl_list"]))
        n_totalunits.append(srl_len)

        if srl_len > 0:
            # print(srl_len)
            # print(sentence_df['srl_list'].unique())
            for i in range(0, srl_len):
                n_element = 0
                # print(i)
                for index, row in sentence_df.iterrows():
                    srl_list = ast.literal_eval(
                        row["srl_list"].replace("I-", "").replace("B-", "")
                    )

                    # print(srl_list)
                    if index == 0:
                        if srl_list[i] != "O":
                            semantic_dic[srl_list[i]] = semantic_dic[srl_list[i]] + 1
                            n_element += 1
                        # print(srl_list[i])
                        previous_srl = srl_list[i]
                    elif srl_list[i] != previous_srl:
                        if srl_list[i] != "O":
                            semantic_dic[srl_list[i]] = semantic_dic[srl_list[i]] + 1
                            n_element += 1
                        # print(srl_list[i])
                        previous_srl = srl_list[i]
                    # print(srl_list[i])

                n_totalelements.append(n_element)

        # print(srl_list)
    u_n_totalunits = sum(n_totalunits)
    try:
        u_md_totalunits = statistics.median(n_totalunits)
    except:
        u_md_totalunits = np.nan
    try:
        u_mx_totalunits = np.percentile(n_totalunits, 95)
    except:
        u_mx_totalunits = np.nan

    u_n_totalelements = sum(n_totalelements)
    try:
        u_md_totalelements = statistics.median(n_totalelements)
    except:
        u_md_totalelements = np.nan
    try:
        u_mx_totalelements = np.percentile(n_totalelements, 95)
    except:
        u_mx_totalelements = np.nan

    try:
        dialog_dic_s = {key: value / u_n_utterance for key, value in dialog_dic.items()}
    except:
        dialog_dic_s = {}

    for index, row in df.iterrows():
        dep_tag = row["dep.dependancy"]
        dependancy_dic[dep_tag] = dependancy_dic[dep_tag] + 1
    try:
        dependancy_dic_w = {
            key: (value / u_n_words) * 100 for key, value in dependancy_dic.items()
        }
    except:
        dependancy_dic_w = {}
    try:
        semantic_dic_u = {
            key: value / u_n_utterance for key, value in semantic_dic.items()
        }
    except:
        semantic_dic_u = {}
    # print(dependancy_dic)
    entry = {}
    entry["filename"] = filename
    entry["u_n_utterance"] = u_n_utterance
    entry["u_n_words"] = u_n_words
    for key in dialog_dic:
        entry["u_n_" + key] = dialog_dic[key]

    for key in dialog_dic_s:
        entry["u_s_" + key] = dialog_dic[key]

    try:
        entry["u_n_statement"] = (
            dialog_dic["statementnonopinion"] + dialog_dic["statementopinion"]
        )
    except:
        pass
    try:
        entry["u_s_statement"] = (
            dialog_dic_s["statementnonopinion"] + dialog_dic_s["statementopinion"]
        )
    except:
        pass

    for key in dependancy_dic:
        entry["u_n_" + key] = dependancy_dic[key]
    for key in dependancy_dic_w:
        entry["u_w_" + key] = dependancy_dic_w[key]
    entry["u_n_totalunits"] = u_n_totalunits
    entry["u_md_totalunits"] = u_md_totalunits
    entry["u_mx_totalunits"] = u_mx_totalunits
    entry["u_n_totalelements"] = u_n_totalelements
    entry["u_md_totalelements"] = u_md_totalelements
    entry["u_mx_totalelements"] = u_mx_totalelements

    for key in semantic_dic:
        entry["u_n_" + key] = semantic_dic[key]

    for key in semantic_dic_u:
        entry["u_s_" + key] = semantic_dic_u[key]

    report_df = pd.concat([report_df, pd.DataFrame([entry])], ignore_index=True)

report_df.to_csv(root_data + feature_folder + "//5_utterance_features.csv")


# In[ ]:


# In[ ]:


# In[ ]:


# In[51]:


sent_parse = dep_parser.predict(sent)
sent_srl = srl_labeler.predict(sentence=sent)
print(sent_srl.keys())
sent_srl["words"]
for i in range(0, len(sent_parse["words"])):
    srl_tag = []
    for j in range(0, len(sent_srl["verbs"])):
        srl_tag.append(sent_srl["verbs"][j]["tags"][i])
    print(
        sent_parse["words"][i],
        sent_parse["pos"][i],
        sent_parse["predicted_dependencies"][i],
        sent_srl["words"][i],
        srl_tag,
    )

len(sent_srl["words"])
print(sent_srl["verbs"])
