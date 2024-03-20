import pandas as pd

#TODO: Clean this up for readability

# This code is originally from 004_WordFeatures
# SENTIMENT DICTIONARIES

def extract_column(df, k):
    d = {}
    for index, row in df.iterrows():
        d[row["Word"].lower()] = row[k]
    return d


def get_me_dic(df)
    # These all have naming idiosyncracies, so they're manually filled in
    l_me_valence = df["sent_valence"].dropna().mean()
    l_me_arousal = df["sent_arousal"].dropna().mean()
    l_me_dominance = df["sent_dominance"].dropna().mean()
    l_me_imagibility = df["glasgow_imagibility"].dropna().mean()
    l_me_gender = df["glasgow_gender"].dropna().mean()
    l_me_size = df["glasgow_size"].dropna().mean()
    l_me_auditory = df["sm_auditory"].dropna().mean()
    l_me_gustatory = df["sm_gustatory"].dropna().mean()
    l_me_haptic = df["sm_haptic"].dropna().mean()
    l_me_interoception = df["sm_interoceptive"].dropna().mean()
    l_me_olfaction = df["sm_olfactory"].dropna().mean()
    l_me_visual = df["sm_visual"].dropna().mean()
    l_me_footleg = df["sm_foot_leg"].dropna().mean()
    l_me_handarm = df["sm_hand_arm"].dropna().mean()
    l_me_head = df["sm_head"].dropna().mean()
    l_me_mouth = df["sm_mouth"].dropna().mean()
    l_me_torso = df["sm_torso"].dropna().mean()
    l_me_sensory = mean(
        [
            l_me_auditory,
            l_me_gustatory,
            l_me_haptic,
            l_me_interoception,
            l_me_olfaction,
            l_me_visual,
        ]
    )
    l_me_motor = mean([l_me_footleg, l_me_handarm, l_me_head, l_me_mouth, l_me_torso])
    try:
        l_me_taboo = df["taboo"].dropna().mean()
    except:
        pass
    me_dic = {
        "l_me_auditory": l_me_auditory,
        "l_me_gustatory": l_me_gustatory,
        "l_me_haptic": l_me_haptic,
        "l_me_interoception": l_me_interoception,
        "l_me_olfaction": l_me_olfaction,
        "l_me_visual": l_me_visual,
        "l_me_footleg": l_me_footleg,
        "l_me_handarm": l_me_handarm,
        "l_me_head": l_me_head,
        "l_me_mouth": l_me_mouth,
        "l_me_torso": l_me_torso,
        "l_me_sensory": l_me_sensory,
        "l_me_motor": l_me_motor,
        "l_me_valence": l_me_valence,
        "l_me_arousal": l_me_arousal,
        "l_me_dominance": l_me_dominance,
        "l_me_imagibility": l_me_imagibility,
        "l_me_gender": l_me_gender,
        "l_me_size": l_me_size,
        "l_me_taboo": l_me_taboo,

    }
    return me_dic

# Sentiment Dictionaries
def get_sent_dics():
    sent_dic = {}
    valence_dic = {}
    arousal_dic = {}
    dominance_dic = {}
    vad_df = pd.read_csv("sentiment_dictionaries//VAD dictionary.csv").drop(
        "Unnamed: 0", axis=1
    )
    vad_df["Word"] = vad_df["Word"].astype(str)
    for index, row in vad_df.iterrows():
        valence_dic[row["Word"].lower()] = row["V.Mean.Sum"]
        arousal_dic[row["Word"].lower()] = row["A.Mean.Sum"]
        dominance_dic[row["Word"].lower()] = row["D.Mean.Sum"]

    with open(
        "sentiment_dictionaries//NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", "r"
    ) as sentiment_doc:
        for each_line in sentiment_doc.readlines():
            # print(each_line.strip().split('\t'))
            each_sent = each_line.strip().split("\t")
            sent_dic[(each_sent[0].lower(), each_sent[1])] = int(each_sent[2])
    return sent_dic, valence_dic, arousal_dic, dominance_dic


# Lexical Dictionaries
def get_lex_dics():
    aoa_df = pd.read_csv("lexical_dictionaries//1_AoA_Prevalence.csv")
    aoa_dic = {}
    aoa_df["Word"] = aoa_df["Word"].astype(str)
    for index, row in aoa_df.iterrows():
        for feature in ["NSyll", "AoA"]:
            aoa_dic[(row["Word"].lower(), feature)] = row[feature]

    semd_df = pd.read_csv("lexical_dictionaries//2_SemD.csv")
    semd_df["item"] = semd_df["item"].astype(str)
    semd_dic = {}
    for index, row in semd_df.iterrows():
        for feature in [
            "mean_cos",
            "SemD",
        ]:  # ,'BNC_wordcount','BNC_contexts','BNC_freq','lg_BNC_freq'
            semd_dic[(row["item"].lower(), feature)] = row[feature]

    calg_conc_dic = {}

    calg_conc = pd.read_excel("lexical_dictionaries//3_CalgaryConcrete.xlsx")
    calg_conc["Word"] = calg_conc["Word"].astype(str)
    calg_conc_dic = extract_column(calg_conc, "WordType")


    rate_conc = pd.read_excel("lexical_dictionaries//4_ConcRate.xlsx")
    rate_conc["Word"] = rate_conc["Word"].astype(str)
    rate_conc_dic = extract_column(rate_conc, "Conc.M")

    prevalence_df = pd.read_excel("lexical_dictionaries//5_prevalence.xlsx")
    prevalence_df["Word"] = prevalence_df["Word"].astype(str)
    prevalence_dic = extract_column(prevalence_df, "Prevalence")

    iconicity_dic = {}
    iconicity_df = pd.read_csv("lexical_dictionaries//6_iconicity.csv")
    iconicity_df["word"] = iconicity_df["word"].astype(str)
    iconicity_dic = extract_column(iconicity_df, "rating")
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
    sensorimotor_dic = {}
    sensorimotor_df = pd.read_csv("lexical_dictionaries//7_sensorimotor.csv")
    for index, row in sensorimotor_df.iterrows():
        for each_item in sensorimotor:
            sensorimotor_dic[(row["Word"].lower(), each_item)] = row[each_item]

    subtlex = ["SUBTLWF", "SUBTLCD"]
    subtlex_dic = {}
    subtlex_df = pd.read_excel("lexical_dictionaries//8_subtlex.xlsx")
    subtlex_df["Word"] = subtlex_df["Word"].astype(str)
    for index, row in subtlex_df.iterrows():
        for each_item in subtlex:
            subtlex_dic[(row["Word"].lower(), each_item)] = row[each_item]

    taboo_df = pd.read_csv("lexical_dictionaries//9_taboo.csv")
    taboo_dic = extract_column(taboo_df, "Taboo")

    glasgow = ["IMAG", "GEND", "SIZE"]
    glasgow_dic = {}
    glasgow_df = pd.read_csv("lexical_dictionaries//10_glasgow.csv")
    glasgow_df["Word"] = subtlex_df["Word"].astype(str)
    for index, row in glasgow_df.iterrows():
        if index != 0:
            for each_item in glasgow:
                glasgow_dic[(row["Words"].lower(), each_item)] = float(row[each_item])
    return (
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
    )
