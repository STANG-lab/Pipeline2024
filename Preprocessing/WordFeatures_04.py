#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import spacy
import ast
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import nltk
from abstract_featurizer import Featurizer
from nltk.corpus import stopwords
from dict_utils import *

# nltk.download("stopwords")
# nltk.download('punkt')
# nltk.download('wordnet')


# function string nan
def isNaN(string):
    return string != string

class WordFeaturizer(Featurizer):

    def __init__(self, outdirs, word_level_csv="4_word_features.csv"):
        super().__init__(outdirs)
        self.word_feature_csv_path = outdirs["Features"] / word_level_csv
        self.word_feat_folder = outdirs["Features/4_word_features"]
        self.get_word_feats()

    def get_word_feats(self):
        swords = stopwords.words("english")
        nlp = spacy.load("en_core_web_sm")
        sensorimotor = [
            "Auditory.mean",
            "Gustatory.mean",
            "Haptic.mean",
            "Interoceptive.mean",
            "Olfactory.mean",
            "Visual.mean",
            "Foot_leg.mean",
            "Hand_arm.mean",
            "Head.mean",
            "Mouth.mean",
            "Torso.mean",
        ]
        emotions = [
            "anger",
            "anticipation",
            "disgust",
            "fear",
            "joy",
            "negative",
            "positive",
            "sadness",
            "surprise",
            "trust",
        ]

        # drive_aggregate_in_path = root_data + word_level_location
        # drive_wordlevel_out_path = root_data + word_feature_location
        #
        # get all the files in the transcripts folder (the folder should just include transcripts)
        all_aggregate_files = self.word_agg.glob("*.csv")


        sent_dics = sent_dic, valence_dic, arousal_dic, dominance_dic = get_sent_dics()
        lex_dics = [
            aoa_dic,
            semd_dic,
            calg_conc_dic,
            rate_conc_dic,
            prevalence_dic,
            iconicity_dic,
            sensorimotor_dic,
            subtlex_dic,
            taboo_dic,
            glasgow_dic,
        ] = get_lex_dics()

        for aggregate in all_aggregate_files:
            # print(aggregate)
            # if aggregate.split("_")[5] in ["SOC", "PIC", "JOU", "BTW", "FLU"]: # This check is useful for some studies
            # print(drive_aggregate_in_path+aggregate)
            df = pd.read_csv(aggregate)
            df["row_id"] = df.index
            df = df.loc[df["is_unintelligible"] == 0].copy()
            df = df.loc[df["is_repetition"] == 0].copy()
            df = df.loc[df["is_partial"] == 0].copy()
            df = df.loc[df["is_punctuation"] == 0].copy()
            df = df.loc[df["is_noise"] == 0].copy()
            df = df.loc[df["is_laugh"] == 0].copy()
            df = df.loc[df["is_filledpause"] == 0].copy()
            df = df[["uid", "speaker", "sentence_id", "token_id", "content", "n_words"]]
            df2 = pd.DataFrame(
                columns=[
                    "uid",
                    "speaker",
                    "sentence_id",
                    "token_id",
                    "content",
                    "wordfeature_id",
                ]
            )
            new_index = 0
            dfs = [df2]
            for index, row in df.iterrows():
                token = row["content"]
                if len(token.split(" ")) == 1:
                    token_split = token.split(" ")
                elif len(token.split(" ")) > 1:
                    token_split = token.replace(".", "").replace(",", "").split(" ")
                for each_token in token_split:
                    dfs.append(pd.DataFrame(
                                    {
                                        "uid": row["uid"],
                                        "speaker": row["speaker"],
                                        "sentence_id": row["sentence_id"],
                                        "token_id": row["token_id"],
                                        "wordfeature_id": new_index,
                                        "content": each_token,
                                    },
                        index=[0]
                            )
                    )
                    new_index += 1
                df2 = pd.concat(dfs, ignore_index=True)
            # lowercase
            df2["word_lower"] = df2["content"].str.lower()

            # tokenized list
            df2["nltk.tokenized"] = df2.apply(
                lambda row: nltk.word_tokenize(str(row["word_lower"]).lower()), axis=1
            )
            df2["nltk.n_token"] = df2.apply(lambda row: len(row["nltk.tokenized"]), axis=1)

            # count stopwords per row:
            df2["nltk.n_stopword"] = df2.apply(
                lambda row: len(list(filter(lambda x: x in swords, row["nltk.tokenized"]))),
                axis=1,
            )

            # df_no_nsv["nltk.n_stopword"] = df_no_nsv.apply(lambda row: row["word_lower"] in swords, axis = 1 )
            df2["is_stopword"] = df2.apply(
                lambda row: 1 if row["nltk.n_stopword"] > 0 else 0, axis=1
            )

            # Create a Tokenizer with the default settings for English
            tokenizer = nlp.tokenizer
            # create tokenization array
            df2["sp.tokenized"] = df2.apply(
                lambda row: list(tokenizer(row["word_lower"]))
                if not isNaN(row["word_lower"])
                else [],
                axis=1,
            )
            # count number of tokens
            df2["sp.n_token"] = df2.apply(lambda row: len(row["sp.tokenized"]), axis=1)

            test_counter = 0

            pos_df = pd.DataFrame(
                columns=[
                    "wordfeature_id",
                    "sp.token_text",
                    "sp.lemma",
                    "sp.pos",
                    "sp.tag",
                    "sp.shape",
                    "sp.is_alpha",
                ]
            )
            pos_dfs = [pos_df]

            for n, l in df2.groupby(["uid", "sentence_id"]):
                # print(l) # grouped df of this sentence
                sent = l["word_lower"].str.cat(
                    sep=" "
                )  # pasting together the original sentence
                # print("SENTENCE: " + sent)
                # do the POS tagging
                doc = nlp(sent)

                # the number of entries in doc should be the same as the length of the tokens
                sum_tokens = sum(l["sp.n_token"])
                assert sum_tokens == len(doc)

                # this loop goes through all the rows that build this sentence. For each row, it goes through the spacy
                # generated array of POS etc and assigns them to this row
                spacy_doc_start_idx = 0
                for idx, r in l.iterrows():
                    n_tok = r["sp.n_token"]

                    doc_slice = doc[spacy_doc_start_idx: spacy_doc_start_idx + n_tok]

                    entry = {
                        "wordfeature_id": idx,
                        "sp.token_text": [t.text for t in doc_slice],
                        "sp.lemma": [t.lemma_ for t in doc_slice],
                        "sp.pos": [t.pos_ for t in doc_slice],
                        "sp.tag": [t.tag_ for t in doc_slice],
                        "sp.shape": [t.shape_ for t in doc_slice],
                        "sp.is_alpha": [t.is_alpha for t in doc_slice],
                    }

                    # moving on
                    spacy_doc_start_idx = spacy_doc_start_idx + n_tok
                    pos_dfs.append(pd.DataFrame([entry]))

                    # pos_df = pd.concat([pos_df, pd.DataFrame([entry])], ignore_index=True)

                    # for debugging to just run over a couple of entries
                    # test_counter = test_counter + 1
                    # if test_counter > 10:
                    if pos_df["sp.lemma"].size == 0:
                        test_counter = test_counter + 1
                        # print('flag: missing tatpicture and aboutyourself', test_counter, df_filt['content'])
                        break
            final_pos_df = pd.concat(pos_dfs, ignore_index=True)

            # merge the parts of speech with the original df
            df2 = df2.merge(
                final_pos_df, how="left", left_on="wordfeature_id", right_on="wordfeature_id"
            )

            # print(df2)
            # sentiments:

            for index, row in df2.iterrows():
                tokens_text = str(row["sp.token_text"])
                token_lema = row["sp.lemma"]
                sent_valence = []
                sent_arousal = []
                sent_dominance = []
                taboo = []
                glasgow_img = []
                glasgow_gend = []
                glasgow_size = []
                aoa_NSyll = []
                aoa_AoA = []
                aoa_Prevalence = []
                semd_SemD = []
                conc_type = []
                conc_rate = []
                prevalence = []
                iconicity = []
                sub_wf = []
                sub_cd = []

                emo_lists = defaultdict(list)
                sm_lists = defaultdict(list)
                def conditional_append(l, dict, key):
                    if dict.get(key, None) is not None:
                        l.append(dict[key])

                for each_token in tokens_text:
                    each_token = each_token.lower()
                    for emo in emotions:
                        emo_lists[emo].append(sent_dic.get((each_token, emo), np.nan))
                    for sm in sensorimotor:
                        sm_lists[sm].append(sensorimotor_dic.get((each_token, sm), np.nan))

                    conditional_append(glasgow_img, glasgow_dic, (each_token, "IMAG"))
                    conditional_append(glasgow_gend, glasgow_dic, (each_token, "GEND"))
                    conditional_append(glasgow_size, glasgow_dic, (each_token, "SIZE"))

                    # try:
                    #     glasgow_img.append(glasgow_dic[(each_token, "IMAG")])
                    #     glasgow_gend.append(glasgow_dic[(each_token, "GEND")])
                    #     glasgow_size.append(glasgow_dic[(each_token, "SIZE")])
                    # except:
                    #     pass

                    conditional_append(sent_valence, taboo_dic, each_token)
                    conditional_append(sent_arousal, arousal_dic, each_token)
                    conditional_append(sent_dominance, dominance_dic, each_token)

                    # try:
                    #     sent_valence.append(valence_dic[each_token])
                    #     sent_arousal.append(arousal_dic[each_token])
                    #     sent_dominance.append(dominance_dic[each_token])
                    # except:
                    #     pass
                    #
                    conditional_append(taboo, taboo_dic, each_token)

                    # try:
                    #     taboo.append(taboo_dic[each_token])
                    # except:
                    #     pass
                    #

                    conditional_append(aoa_NSyll, aoa_dic, (each_token, "NSyll"))
                    conditional_append(aoa_AoA, aoa_dic, (each_token, "AoA") )

                    # try:
                    #     aoa_NSyll.append(aoa_dic[each_token, "NSyll"])
                    #     aoa_AoA.append(aoa_dic[each_token, "AoA"])
                    # except:
                    #     pass

                    conditional_append(semd_SemD, semd_dic, (each_token, "SemD"))
                    # try:
                    #     semd_SemD.append(semd_dic[each_token, "SemD"])
                    # except:
                    #     pass

                    conditional_append(conc_type, calg_conc_dic, each_token)
                    # try:
                    #     conc_type.append(calg_conc_dic[each_token])
                    # except:
                    #     pass

                    conditional_append(conc_rate, rate_conc_dic, each_token)

                    # try:
                    #     conc_rate.append(rate_conc_dic[each_token])
                    # except:
                    #     pass

                    conditional_append(prevalence, prevalence_dic, each_token)
                    # try:
                    #     prevalence.append(prevalence_dic[each_token])
                    # except:
                    #     pass

                    conditional_append(iconicity, iconicity_dic, each_token)
                    # try:
                    #     iconicity.append(iconicity_dic[each_token])
                    # except:
                    #     pass
                    conditional_append(sub_wf, subtlex_dic, (each_token, "SUBTLWF"))
                    conditional_append(sub_cd, subtlex_dic, (each_token, "SUBTLCD"))

                    # try:
                    #     sub_wf.append(subtlex_dic[(each_token, "SUBTLWF")])
                    #     sub_cd.append(subtlex_dic[(each_token, "SUBTLCD")])
                    # except:
                    #     pass

                df2.at[index, "conc_type"] = str(conc_type)
                for emo in emotions:
                    df2.at[index, f"sent_{emo}"] = mean(emo_lists[emo])
                for sm in sensorimotor:
                    df2.at[index, f"sm_{sm.lower().split('.')[0]}"] = mean(
                        sm_lists[sm]
                    )
                def conditional_mean(l, key):
                    if df2.get(index, key) is not None and len(l) > 0:
                        df2.at[index, key] = mean(l)


                conditional_mean(sent_valence, "sent_valence")
                conditional_mean(sent_arousal,  "sent_arousal")
                conditional_mean(sent_dominance, "sent_dominance")

                # try:
                #     df2.at[index, "sent_valence"] = mean(sent_valence)
                #     df2.at[index, "sent_arousal"] = mean(sent_arousal)
                #     df2.at[index, "sent_dominance"] = mean(sent_dominance)
                # except:
                #     pass
                conditional_mean(aoa_NSyll, "aoa_NSyll")
                conditional_mean(aoa_AoA, "aoa_AoA")
                # try:
                #     df2.at[index, "aoa_NSyll"] = mean(aoa_NSyll)
                #     df2.at[index, "aoa_AoA"] = mean(aoa_AoA)
                # except:
                #     pass
                conditional_mean(semd_SemD, "semd_SemD")
                # try:
                #     df2.at[index, "semd_SemD"] = mean(semd_SemD)
                # except:
                #     pass
                conditional_mean(conc_rate, "conc_rate")
                try:
                    df2.at[index, "conc_rate"] = mean(conc_rate)
                except:
                    pass
                try:
                    df2.at[index, "prevalence"] = mean(prevalence)
                except:
                    pass
                try:
                    df2.at[index, "iconicity"] = mean(iconicity)
                except:
                    pass
                try:
                    df2.at[index, "subtlex_wf"] = mean(sub_wf)
                    df2.at[index, "subtlex_cd"] = mean(sub_cd)
                except:
                    pass
                try:
                    df2.at[index, "taboo"] = mean(taboo)
                except:
                    pass
                try:
                    df2.at[index, "glasgow_imagibility"] = mean(glasgow_img)
                    df2.at[index, "glasgow_gender"] = mean(glasgow_gend)
                    df2.at[index, "glasgow_size"] = mean(glasgow_size)
                except:
                    pass
            df2.to_csv(self.word_feat_folder / Path(aggregate).name)
            #
            # [['uid', 'speaker','sentence_id', 'token_id', 'content', 'wordfeature_id',
            # 'word_lower','nltk.tokenized','nltk.n_token','nltk.n_stopword','is_stopword',
            #  'sp.tokenized','sp.n_token','sp.token_text','sp.lemma','sp.pos','sp.tag','sp.shape','sp.is_alpha',
        #     'aoa_NSyll','aoa_AoA',
        #    'semd_SemD',
        #  'sent_valence','sent_arousal','sent_dominance',
        #  'sent_anger','sent_anticipation','sent_disgust','sent_fear','sent_joy','sent_negative','sent_positive','sent_sadness','sent_surprise','sent_trust']].to_csv(drive_wordlevel_out_path+aggregate)

        report_df = pd.DataFrame(
            columns=[
                "filename",
            ]
        )
        reports = []

        for aggregate in self.word_feat_folder.glob("*.csv"):
            df = pd.read_csv(aggregate, sep=',')
            df = df.loc[df["speaker"] == "Subject:"].copy()
            # df = df.loc[df["speaker"] == "SPEAKER_01"].copy()

            l_n_words = len(df)
            def l_n_norm(x):
                try:
                    return df[x].sum() / max(1, l_n_words) * 100
                except:
                    return np.nan

            l_w_emotions = {f"l_w_{emo}": l_n_norm(f"sent_{emo}") for emo in emotions}

            abs_conc_counter = Counter()
            pos_counter = Counter()

            for index, row in df.iterrows():
                conc_list = ast.literal_eval(row["conc_type"])
                abs_conc_counter.update(conc_list)
                pos_list = ast.literal_eval(row["sp.pos"])
                pos_counter.update(pos_list)

            pos_counter.update(
                {key.lower(): abs_conc_counter[key] for key in abs_conc_counter.keys()}
            )
            l_n_dict = pos_counter
            l_n_dict["stopwords"] = df["is_stopword"].sum()
            l_n_dict["syllable"] = df["aoa_NSyll"].sum()

            l_mx_syllable = df["aoa_NSyll"].dropna().quantile(0.95)
            l_mx_aoa = df["aoa_AoA"].dropna().quantile(0.95)
            l_mx_dic = {
                "l_mx_syllable": l_mx_syllable,
                "l_mx_aoa": l_mx_aoa,
            }

            l_md_aoa = df["aoa_AoA"].dropna().median()
            l_md_semd = df["semd_SemD"].dropna().median()
            l_md_concrete = df["conc_rate"].dropna().median()
            l_md_prevalence = df["prevalence"].dropna().median()
            l_md_wordfreq = df["subtlex_wf"].dropna().median()
            l_md_contextdiversity = df["subtlex_cd"].dropna().median()

            l_md_dic = {
                "l_md_aoa": l_md_aoa,
                "l_md_semd": l_md_semd,
                "l_md_prevalence": l_md_prevalence,
                "l_md_concrete": l_md_concrete,
                "l_md_wordfreq": l_md_wordfreq,
                "l_md_contextdiversity": l_md_contextdiversity,
            }
            l_me_dic = get_me_dic(df)

            # Enforce appropriate key formats
            l_w_dict = {f"l_w_{key}": l_n_norm(l_n_dict[key]) for key in l_n_dict.keys()}
            l_w_dict.update(l_w_emotions)
            l_n_dict["words"] = l_n_words
            l_n_dict = {f"l_n_{key}": l_n_dict[key] for key in l_n_dict.keys()}
            # each_report = pd.DataFrame(l_n_dict | l_w_dict | l_me_dic | l_md_dic | l_mx_dic, index=[0])
            each_report = pd.DataFrame({**l_n_dict, **l_w_dict, **l_me_dic, **l_md_dic, **l_mx_dic}, index=[0])
            reports.append(each_report)
        report_df = pd.concat(reports, ignore_index=True)
        # report_df = pd.concat(reports, index=[0])
        report_df.to_csv(self.word_feature_csv_path)


# In[109]:


# report_df


# In[106]:

# each_report = pd.DataFrame(
#     [
#         {
#             "filename": filename,
#             "l_n_words": l_n_words,
#             "l_n_stopwords": l_n_stopwords,
#             "l_w_stopwords": l_w_stopwords,
#             "l_n_syllable": l_n_syllable,
#             "l_w_syllable": l_w_syllable,
#             "l_mx_syllable": l_mx_syllable,
#             "l_md_aoa": l_md_aoa,
#             "l_mx_aoa": l_mx_aoa,
#             "l_md_semd": l_md_semd,
#             "l_md_concrete": l_md_concrete,
#             "l_md_prevalence": l_md_prevalence,
#             "l_me_auditory": l_me_auditory,
#             "l_me_gustatory": l_me_gustatory,
#             "l_me_haptic": l_me_haptic,
#             "l_me_interoception": l_me_interoception,
#             "l_me_olfaction": l_me_olfaction,
#             "l_me_visual": l_me_visual,
#             "l_me_footleg": l_me_footleg,
#             "l_me_handarm": l_me_handarm,
#             "l_me_head": l_me_head,
#             "l_me_mouth": l_me_mouth,
#             "l_me_torso": l_me_torso,
#             "l_me_sensory": l_me_sensory,
#             "l_me_motor": l_me_motor,
#             "l_md_wordfreq": l_md_wordfreq,
#             "l_md_contextdiversity": l_md_contextdiversity,
#             "l_w_anger": l_w_anger,
#             "l_w_anticipation": l_w_anticipation,
#             "l_w_disgust": l_w_disgust,
#             "l_w_fear": l_w_fear,
#             "l_w_joy": l_w_joy,
#             "l_w_negative": l_w_negative,
#             "l_w_positive": l_w_positive,
#             "l_w_sadness": l_w_sadness,
#             "l_w_surprise": l_w_surprise,
#             "l_w_trust": l_w_trust,
#             "l_me_valence": l_me_valence,
#             "l_me_arousal": l_me_arousal,
#             "l_me_dominance": l_me_dominance,
#             "l_me_imagibility": l_me_imagibility,
#             "l_me_gender": l_me_gender,
#             "l_me_size": l_me_size,
#             "l_me_taboo": l_me_taboo,
#             "l_n_concrete": l_n_concrete,
#             "l_n_abstract": l_n_abstract,
#             "l_n_INTJ": l_n_INTJ,
#             "l_n_PROPN": l_n_PROPN,
#             "l_n_NUM": l_n_NUM,
#             "l_n_VERB": l_n_VERB,
#             "l_n_NOUN": l_n_NOUN,
#             "l_n_ADV": l_n_ADV,
#             "l_n_ADJ": l_n_ADJ,
#             "l_n_ADP": l_n_ADP,
#             "l_n_PRON": l_n_PRON,
#             "l_n_AUX": l_n_AUX,
#             "l_n_DET": l_n_DET,
#             "l_n_SCONJ": l_n_SCONJ,
#             "l_n_PART": l_n_PART,
#             "l_n_CCONJ": l_n_CCONJ,
#             "l_n_X": l_n_X,
#             "l_w_concrete": l_w_concrete,
#             "l_w_abstract": l_w_abstract,
#             "l_w_INTJ": l_w_INTJ,
#             "l_w_PROPN": l_w_PROPN,
#             "l_w_NUM": l_w_NUM,
#             "l_w_VERB": l_w_VERB,
#             "l_w_NOUN": l_w_NOUN,
#             "l_w_ADV": l_w_ADV,
#             "l_w_ADJ": l_w_ADJ,
#             "l_w_ADP": l_w_ADP,
#             "l_w_PRON": l_w_PRON,
#             "l_w_AUX": l_w_AUX,
#             "l_w_DET": l_w_DET,
#             "l_w_SCONJ": l_w_SCONJ,
#             "l_w_PART": l_w_PART,
#             "l_w_CCONJ": l_w_CCONJ,
#             "l_w_X": l_w_X,
#         }
#     ]
# )

#     "ADJ": "adjective",
#     "ADP": "adposition",
#     "ADV": "adverb",
#     "AUX": "auxiliary",
#     "CONJ": "conjunction",
#     "CCONJ": "coordinating conjunction",
#     "DET": "determiner",
#     "INTJ": "interjection",
#     "NOUN": "noun",
#     "NUM": "numeral",
#     "PART": "particle",
#     "PRON": "pronoun",
#     "PROPN": "proper noun",
#     "PUNCT": "punctuation",
#     "SCONJ": "subordinating conjunction",
#     "SYM": "symbol",
#     "VERB": "verb",
#     "X": "other",
#     "EOL": "end of line",
#     "SPACE": "space",

# aoa_dic, semd_df,
# calg_conc_dic, rate_conc_dic,
# prevalence_dic, iconicity_dic, sensorimotor_dic, subtlex_dic, taboo_dic, glasgow_dic
