#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from allennlp_models import pretrained
from somajo import SoMaJo
from dialog_tag import DialogTag
import torch
from transformers import BertModel, BertTokenizer
from Bio import pairwise2
from allennlp_models import pretrained
import os

coref_labeler = pretrained.load_predictor("coref-spanbert")
srl_labeler = pretrained.load_predictor("structured-prediction-srl-bert")
dep_parser = pretrained.load_predictor("structured-prediction-biaffine-parser")
speechact_model = DialogTag("distilbert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# In[4]:


def srl_coref_embed(text):
    # Sentence Tokenizer
    sentence_df = pd.DataFrame()
    tokenizer = SoMaJo("en_PTB", split_camel_case=True)
    somajo_sentences = tokenizer.tokenize_text([text])
    n_sent = 0
    for sentence in somajo_sentences:
        each_sentence = []
        for token in sentence:
            each_sentence.append(token.text)
        content_sent = " ".join(each_sentence).strip()
        instance = pd.DataFrame(
            {"sentence_id": [n_sent], "content": [content_sent.strip()]}
        )
        sentence_df = pd.concat([sentence_df, instance], ignore_index=True)
        n_sent += 1
    for index, row in sentence_df.iterrows():
        sentence_df.at[index, "dialogue_tag"] = speechact_model.predict_tag(
            row["content"]
        )
    ###SRL and Coref
    token_index_srl = pd.DataFrame(
        columns=[
            "dialogue_tag",
            "token_index",
            "token_index_utterance",
            "utterance_index",
            "token_srl",
            "all_roles",
        ]
    )
    n1 = 0
    document = []
    n3 = 0
    all_srl = []
    for index, row in sentence_df.iterrows():
        utterance = row["content"]
        speech_act = row["dialogue_tag"]
        document.append(utterance)
        n2 = 0
        srl_pred = srl_labeler.predict(sentence=utterance)
        all_srl.append(srl_pred)
        token_index_srl_utterance = pd.DataFrame(
            columns=[
                "token_index",
                "token_index_utterance",
                "utterance_index",
                "token_srl",
                "all_roles",
            ]
        )
        for each_token in srl_pred["words"]:
            # token_index_srl_utterance.at[n2, 'uid'] = subject
            # token_index_srl_utterance.at[n2, 'task'] = task
            token_index_srl_utterance.at[n2, "dialogue_tag"] = speech_act
            token_index_srl_utterance.at[n2, "token_index"] = n1
            token_index_srl_utterance.at[n2, "token_index_utterance"] = n2
            token_index_srl_utterance.at[n2, "utterance_index"] = n3
            token_index_srl_utterance.at[n2, "token_srl"] = each_token
            all_roles = []
            for each_predicate in srl_pred["verbs"]:
                # print(each_predicate)
                try:
                    all_roles.append(each_predicate["tags"][n2].split("-", 1)[1])
                except:
                    all_roles.append(each_predicate["tags"][n2])
                # verb_tuple = (each_predicate['verb'], each_predicate['tags'][n2])
                # all_roles.append(verb_tuple)
            token_index_srl_utterance.at[n2, "all_roles"] = all_roles
            n2 = n2 + 1
            n1 = n1 + 1
        token_index_srl = pd.concat(
            [token_index_srl, token_index_srl_utterance], ignore_index=True
        )
        n3 = n3 + 1
    document = " ".join(document)

    coref_pred = coref_labeler.predict(document=document)
    ##
    clusters = coref_pred["clusters"]
    tokens = coref_pred["document"]
    n = 0
    token_index_coref = pd.DataFrame(
        columns=["token_index", "token_coref", "n_clusters", "clusters"]
    )
    for each_token in tokens:
        token_index_coref.at[n, "token_index"] = n
        token_index_coref.at[n, "token_coref"] = each_token
        n = n + 1

    n = 0
    for each_cluster in clusters:
        # print(f'cluster_{n}')
        # print(each_cluster)
        for each_element in each_cluster:
            # print(each_element)
            for token_index in range(each_element[0], each_element[1] + 1):
                # print(token_index)
                if token_index_coref.isnull().at[token_index, "clusters"]:
                    token_index_coref.at[token_index, "clusters"] = f"cluster_{n}"
                else:
                    token_index_coref.at[token_index, "clusters"] = (
                        token_index_coref.at[token_index, "clusters"]
                        + ", "
                        + f"cluster_{n}"
                    )
        n += 1
    for index, row in token_index_coref.iterrows():
        try:
            n_clusters = len(row["clusters"].split(","))
            token_index_coref.at[index, "n_clusters"] = n_clusters
        except:
            pass
    # token_index_coref.head(10)
    ##
    coref_srl = token_index_srl.merge(token_index_coref, on="token_index")

    # bert_pred = bert_tokenizer.tokenize("[CLS] " + document + " [SEP]")
    # n = 0
    # token_index_bert = pd.DataFrame(columns=['token_index', 'token', 'bert_embedding'])
    # for each_token in bert_pred:
    #    token_index_bert.at[n, 'token_index'] = n
    #   token_index_bert.at[n, 'token'] = each_token
    #  n = n+1

    n = 0
    for index, row in sentence_df.iterrows():
        document = row["content"]
        bert_pred = bert_tokenizer.tokenize(document)

        token_index_bert = pd.DataFrame(
            columns=["token_index", "token", "bert_embedding"]
        )
        for each_token in bert_pred:
            token_index_bert.at[n, "token_index"] = n
            token_index_bert.at[n, "token"] = each_token
            n = n + 1
        sentence_df.at[index, "n_tokens_bert"] = n
        sentence_df.at[index, "bert_batch"] = int(n / 510)

    def bert_text_preparation(text, bert_tokenizer):
        """
        Preprocesses text input in a way that BERT can interpret.
        """
        marked_text = "[CLS] " + text + " [SEP]"  # add special tokens

        # BERT uses a subword tokenizer (WordPiece),
        # so the maximum length corresponds to 512 subword tokens.
        tokenized_text = bert_tokenizer.tokenize(marked_text)[:512]  # BERT max 512

        indexed_tokens = bert_tokenizer.convert_tokens_to_ids(
            tokenized_text
        )  # find token IDs
        segments_ids = [1] * len(
            indexed_tokens
        )  # for formating and vectors matrix calculations

        # convert inputs to tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensor = torch.tensor([segments_ids])

        return tokenized_text, tokens_tensor, segments_tensor

    def get_bert_embeddings(tokens_tensor, segments_tensor, model):
        """
        Obtains BERT embeddings for tokens, in context of the given response.
        """
        # gradient calculation id disabled
        with torch.no_grad():
            # obtain hidden states
            outputs = model(tokens_tensor, segments_tensor)
            hidden_states = outputs[2]

        # concatenate the tensors for all layers
        # use "stack" to create new dimension in tensor
        token_embeddings = torch.stack(hidden_states, dim=0)

        # remove dimension 1, the "batches"
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        # swap dimensions 0 and 1 so we can loop over tokens
        token_embeddings = token_embeddings.permute(1, 0, 2)

        # intialized list to store embeddings
        token_vecs_sum = []

        # "token_embeddings" is a [Y x 12 x 768] tensor
        # where Y is the number of tokens in the response

        # loop over tokens in response
        for token in token_embeddings:
            # "token" is a [12 x 768] tensor

            # sum the vectors from the last four layers
            sum_vec = torch.sum(token[-4:], dim=0)
            token_vecs_sum.append(sum_vec)

        return token_vecs_sum

    token_index_bert = pd.DataFrame(columns=["token_index", "token", "bert_embedding"])
    m = 0
    for each_batch in sentence_df["bert_batch"].unique():
        token_index_bert_batch = pd.DataFrame(
            columns=["token_index", "token", "bert_embedding"]
        )
        n = 0
        document = []
        for index, row in sentence_df.loc[
            sentence_df["bert_batch"] == each_batch
        ].iterrows():
            document.append(row["content"])
        document = " ".join(document)
        bert_pred = bert_tokenizer.tokenize("[CLS] " + document + " [SEP]")
        for each_token in bert_pred:
            token_index_bert_batch.at[n, "token_index"] = m
            token_index_bert_batch.at[n, "token"] = each_token
            m = m + 1
            n = n + 1
        tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(
            document, bert_tokenizer
        )
        list_token_embeddings = get_bert_embeddings(
            tokens_tensor, segments_tensors, model
        )
        for i in range(0, len(list_token_embeddings)):
            token_index_bert_batch.at[i, "bert_embedding"] = list(
                list_token_embeddings[i].numpy()
            )
        token_index_bert = pd.concat(
            [token_index_bert, token_index_bert_batch], ignore_index=True
        )

    # tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(document, bert_tokenizer)
    # list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)
    # for i in range(0, len(list_token_embeddings)):
    #  token_index_bert.at[i, 'bert_embedding'] = list_token_embeddings[i]

    # Get the token sequences from the dataframes
    seq1 = coref_srl["token_srl"].tolist()
    seq2 = token_index_bert["token"].tolist()

    srl_tokens_str = " ".join(seq1).lower()
    embed_tokens_str = " ".join(seq2).lower()

    # Perform sequence alignment using the Needleman-Wunsch algorithm
    alignments = pairwise2.align.globalxx(srl_tokens_str, embed_tokens_str)

    # Best alignment
    best_alignment = alignments[0]
    sequence_alignment = pairwise2.format_alignment(*best_alignment)

    i = 0

    for index, row in coref_srl.iterrows():
        each_token = row["token_srl"].lower()
        # print('-')
        for i in range(i, len(best_alignment[0])):
            if best_alignment[0][i] == each_token[0]:
                coref_srl.at[index, "token_start"] = i
                # print(each_token, best_alignment[0][i],i, each_token[0],0)
                k = i
                for j in range(0, len(each_token)):
                    for k in range(k, len(best_alignment[0])):
                        if best_alignment[0][k] == each_token[j]:
                            # print(each_token, best_alignment[0][k],k, each_token[j],j)
                            coref_srl.at[index, "token_end"] = k
                            k = i + j + 1
                            break
                # srl.at[index, 'token_end'] = k

                i = k
                break

    i = 0

    for index, row in token_index_bert.iterrows():
        each_token = row["token"].lower()
        # print('-')
        for i in range(i, len(best_alignment[1])):
            if best_alignment[1][i] == each_token[0]:
                token_index_bert.at[index, "token_start"] = i
                # print(each_token, best_alignment[0][i],i, each_token[0],0)
                k = i
                for j in range(0, len(each_token)):
                    for k in range(k, len(best_alignment[1])):
                        if best_alignment[1][k] == each_token[j]:
                            # print(each_token, best_alignment[0][k],k, each_token[j],j)
                            token_index_bert.at[index, "token_end"] = k
                            k = i + j + 1
                            break
                # srl.at[index, 'token_end'] = k

                i = k
                break
    coref_srl["bert_embedding"] = ""
    for index, row in coref_srl.iterrows():
        embedding = []
        bert_tokens = []
        for index2, row2 in token_index_bert.iterrows():
            if (
                row2["token_start"] >= row["token_start"]
                and row2["token_end"] <= row["token_end"]
            ):
                bert_tokens.append(row2["token"])
                embedding.append(row2["bert_embedding"])
        try:
            coref_srl.at[index, "bert_token"] = bert_tokens
        except:
            coref_srl.at[index, "bert_token"] = str(bert_tokens)
        coref_srl.at[index, "bert_embedding"] = embedding
        # coref_srl.drop(['token_start'], axis=1, inplace=True)
    return (
        sentence_df,
        coref_srl.drop(["token_start", "token_end"], axis=1),
        sequence_alignment,
    )


# In[5]:


### Points to the folder that contains all data
root_data = "C://Users//ANikzad//Desktop//Local_Pipeline//Data//"

### Specify Source Folder
# --- Remora
word_level_location = root_data + "Remora-2023//Batch-1//3_word_aggregates//"

### Specify Destination folder
# --- Remora
dicourse_feature_location_semgraph = (
    root_data
    + "Remora-2023//Batch-1//6_discourse_level_features//6b_graphs//6b2_semgraphs//raw_df//"
)


# In[6]:


# get all the files in the transcripts folder (the folder should just include transcripts)
all_aggregate_files = os.listdir(word_level_location)
len(all_aggregate_files)


# In[7]:


speakers = {"Subject": "Participant", "Interviewer": "Interviewer", "Other": "Other"}
for aggregate in all_aggregate_files:
    # if aggregate.split('_')[5] in [ 'JOU', 'SOC', 'PIC', 'BTW']:
    if aggregate.split(".")[0] == "remora_9394_T1_9394T132_v2023_JOU_HowsItGoing":
        # print(aggregate.split('_')[5])
        # print(drive_aggregate_in_path+aggregate)
        df = pd.read_csv(word_level_location + aggregate)
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
        ].reset_index()
        all_turns = []
        turn_content = []
        for index, row in df.iterrows():
            if index == 0:
                current_speaker = row["speaker"]
                turn_content.append(row["content"])
            else:
                current_speaker = row["speaker"]
                previous_speaker = df.at[index - 1, "speaker"]
                if current_speaker == previous_speaker:
                    turn_content.append(row["content"])
                else:
                    all_turns.append(
                        [
                            speakers[previous_speaker],
                            " ".join(turn_content)
                            .replace("  ", " ")
                            .replace(" .", ".")
                            .replace(" !", "!")
                            .replace(" ,", ",")
                            .replace(" ?", "?")
                            .replace("#", "")
                            .replace(" ' ", "'")
                            .replace(" '", "'")
                            .replace("  ", " "),
                        ]
                    )
                    turn_content = []
                    turn_content.append(row["content"])
        all_turns.append(
            [
                speakers[current_speaker],
                " ".join(turn_content)
                .replace("  ", " ")
                .replace(" .", ".")
                .replace(" !", "!")
                .replace(" ,", ",")
                .replace(" ?", "?")
                .replace("#", "")
                .replace(" ' ", "'")
                .replace(" '", "'")
                .replace("  ", " "),
            ]
        )
        turn_df = pd.DataFrame(all_turns, columns=["speaker", "turn_content"])
        sample = []
        for index, row in turn_df.iterrows():
            sample.append(row["speaker"] + ": " + row["turn_content"])
        sample = "\n".join(sample)
        sentence, tokens, alignment = srl_coref_embed(sample)
        final_df = pd.DataFrame()
        for index, row in tokens.iterrows():
            if index + 1 < len(tokens):
                if tokens.at[index + 1, "token_coref"] == ":":
                    current_speaker = row["token_coref"]
                elif row["token_coref"] not in [":", "\n"]:
                    entry = {}
                    entry["speaker"] = current_speaker
                    entry["token_index"] = row["token_index"]
                    entry["token_index_utterance"] = row["token_index_utterance"]
                    entry["utterance_index"] = row["utterance_index"]
                    entry["dialogue_tag"] = row["dialogue_tag"]
                    entry["token_srl"] = row["token_srl"]
                    entry["all_roles"] = row["all_roles"]
                    entry["token_coref"] = row["token_coref"]
                    entry["n_clusters"] = row["n_clusters"]
                    entry["clusters"] = row["clusters"]
                    entry["bert_embedding"] = row["bert_embedding"]
                    entry["bert_token"] = row["bert_token"]
                    final_df = pd.concat(
                        [final_df, pd.DataFrame([entry])], ignore_index=True
                    )
            else:
                entry = {}
                entry["speaker"] = current_speaker
                entry["token_index"] = row["token_index"]
                entry["token_index_utterance"] = row["token_index_utterance"]
                entry["utterance_index"] = row["utterance_index"]
                entry["dialogue_tag"] = row["dialogue_tag"]
                entry["token_srl"] = row["token_srl"]
                entry["all_roles"] = row["all_roles"]
                entry["token_coref"] = row["token_coref"]
                entry["n_clusters"] = row["n_clusters"]
                entry["clusters"] = row["clusters"]
                entry["bert_embedding"] = row["bert_embedding"]
                entry["bert_token"] = row["bert_token"]
                final_df = pd.concat(
                    [final_df, pd.DataFrame([entry])], ignore_index=True
                )
        final_df.to_excel(
            dicourse_feature_location_semgraph + aggregate.split(".")[0] + ".xlsx"
        )


# In[9]:


all_turns


# In[8]:


final_df
