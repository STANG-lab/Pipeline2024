#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import random
import os
import string
from spacy.lang.en import English

nlp = English()
import spacy

nlp = spacy.load("en_core_web_sm")
import networkx as nx
from statistics import mean
from statistics import stdev


# In[4]:


### Points to the folder that contains all data
root_data = "C://Users//ANikzad//Desktop//Local_Pipeline//Data//"

### Specify Source Folder
# --- Remora
word_level_location = "Remora-2023//Batch-1//3_word_aggregates//"

### Specify Destination folder
# --- Remora
dicourse_feature_location = "Remora-2023//Batch-1//6_discourse_level_features//"

drive_aggregate_in_path = root_data + word_level_location
drive_discourselevel_graph_out_path = (
    root_data + dicourse_feature_location + "6b_graphs//6b1_strgraphs//"
)


# In[5]:


# get all the files in the transcripts folder (the folder should just include transcripts)
all_aggregate_files = os.listdir(drive_aggregate_in_path)
len(all_aggregate_files)


# In[6]:


### graph feature function ###
def graph_features(aggregate, G_seq):
    entry = {}
    entry["uid"] = aggregate.split(".")[0]
    entry["g_sq_nodes"] = G_seq.number_of_nodes()
    entry["g_sq_edges"] = G_seq.number_of_edges()

    try:
        entry["g_sq_degree"] = sum([d for (n, d) in nx.degree(G_seq)]) / float(
            G_seq.number_of_nodes()
        )
    except:
        entry["g_sq_degree"] = 0
    try:
        entry["g_sq_density"] = nx.density(G_seq)
    except:
        entry["g_sq_density"] = 0
    try:
        entry["g_sq_diameter"] = nx.diameter(nx.to_undirected(G_seq))
    except:
        entry["g_sq_diameter"] = 0
    try:
        entry["g_sq_aspl"] = nx.average_shortest_path_length(nx.to_undirected(G_seq))
    except:
        pass
    try:
        entry["g_sq_lscc"] = len(max(nx.strongly_connected_components(G_seq), key=len))
    except:
        pass

    parallel_edge = 0
    l2 = 0
    l3 = 0
    for node1 in nx.nodes(G_seq):
        # print(node1)
        for node2 in nx.neighbors(G_seq, node1):
            # print('\t', node2)
            parallel_edge = parallel_edge + (
                G_seq.number_of_edges(u=node1, v=node2) - 1
            )
            if node1 in nx.neighbors(G_seq, node2):
                l2 += 1
            for node3 in nx.neighbors(G_seq, node2):
                if node1 in nx.neighbors(G_seq, node3):
                    if node1 != node2 and node2 != node3:
                        l3 += 1
    entry["g_sq_pe"] = parallel_edge
    entry["g_sq_l1"] = nx.number_of_selfloops(G_seq)
    entry["g_sq_l2"] = l2 / 2
    entry["g_sq_l3"] = l3 / 3

    G_seq2 = nx.Graph()
    for u, v, data in G_seq.edges(data=True):
        w = data["weight"] if "weight" in data else 1.0
        if G_seq2.has_edge(u, v):
            G_seq2[u][v]["weight"] += w
        else:
            G_seq2.add_edge(u, v, weight=w)
    try:
        entry["g_sq_cc"] = nx.average_clustering(G_seq2.to_directed())
    except:
        pass
    try:
        entry["g_sq_largestclique"] = nx.approximation.large_clique_size(G_seq2)
    except:
        pass

    diameter_rand = []
    aspl_rand = []
    lscc_rand = []
    # lcc_rand = []
    # ncc_rand = []
    cc_rand = []

    for seedidx in range(0, 200):
        seed = random.randint(24, 5000)  # Ensures random selection of seeded graphs
        G_rand = nx.gnm_random_graph(
            entry["g_sq_nodes"], entry["g_sq_edges"], seed=seed, directed=True
        )
        try:
            diameter_rand.append(nx.diameter(nx.to_undirected((G_rand))))
        except:
            pass
        try:
            cc_rand.append(nx.average_clustering(G_rand))
        except:
            pass
        try:
            aspl_rand.append(nx.average_shortest_path_length(nx.to_undirected(G_rand)))
        except:
            pass
        try:
            lscc_rand.append(
                len(max(nx.strongly_connected_components(G_rand), key=len))
            )
        except:
            pass
    try:
        entry["g_sq_zlscc"] = (entry["g_sq_lscc"] - mean(lscc_rand)) / stdev(lscc_rand)
    except:
        pass
    try:
        entry["g_sq_zdiameter"] = (entry["g_sq_diameter"] - mean(lscc_rand)) / stdev(
            lscc_rand
        )
    except:
        pass
    try:
        entry["g_sq_zaspl"] = (entry["g_sq_aspl"] - mean(lscc_rand)) / stdev(lscc_rand)
    except:
        pass
    try:
        entry["g_sq_zcc"] = (entry["g_sq_cc"] - mean(lscc_rand)) / stdev(lscc_rand)
    except:
        pass
    return entry


# In[7]:


### structural graph features
df_strgfeatures = pd.DataFrame()
for aggregate in all_aggregate_files:
    if aggregate.split("_")[5] in ["JOU", "SOC", "PIC"]:
        # print(aggregate.split('_')[5])
        # print(drive_aggregate_in_path+aggregate)
        df = pd.read_csv(drive_aggregate_in_path + aggregate)
        df["row_id"] = df.index
        df = df.loc[df["is_unintelligible"] == 0].copy()
        # df = df.loc[df['is_repetition'] == 0].copy()
        df = df.loc[df["is_partial"] == 0].copy()
        df = df.loc[df["is_punctuation"] == 0].copy()
        df = df.loc[df["is_noise"] == 0].copy()
        df = df.loc[df["is_laugh"] == 0].copy()
        df = df.loc[df["is_filledpause"] == 0].copy()

        # df = df[['uid', 'speaker','sentence_id', 'token_id', 'content',]]
        df = (
            df.loc[df["speaker"] == "Subject"]
            .copy()
            .reset_index()[["sentence_id", "token_id", "content"]]
        )
        node_sequence = []
        for index, row in df.iterrows():
            content = row["content"]
            content_nlp = nlp(content)
            for each_nlp in content_nlp:
                # print(row['token_id'], row['sentence_id'],each_nlp.lemma_)
                node_sequence.append(each_nlp.lemma_)
        node_sequence = " ".join(node_sequence).replace(" - ", "-").split(" ")
        with open(
            drive_discourselevel_graph_out_path + aggregate.split(".")[0] + ".txt", "w"
        ) as node_list:
            for each_node in node_sequence:
                node_list.write(each_node)
                node_list.write("\n")
        G_seq = nx.MultiDiGraph()

        for i in range(0, len(node_sequence) - 1):
            G_seq.add_edge(node_sequence[i], node_sequence[i + 1], weight=1)

        entry = graph_features(aggregate, G_seq)

        dynamic_entry = []
        m = 0
        if len(node_sequence) > 40:
            # print(len(node_sequence))
            while m < (len(node_sequence) - 30):
                window_nodes = node_sequence[m : m + 30]
                m = m + 10
                G_seq30 = nx.MultiDiGraph()
                for i in range(0, len(window_nodes) - 1):
                    G_seq30.add_edge(window_nodes[i], window_nodes[i + 1], weight=1)
                dynamic_entry.append(graph_features(aggregate, G_seq30))
            if m != (len(node_sequence) - 1):
                window_nodes = node_sequence[-30:]
                G_seq30 = nx.MultiDiGraph()
                for i in range(0, len(window_nodes) - 1):
                    G_seq30.add_edge(window_nodes[i], window_nodes[i + 1], weight=1)
                dynamic_entry.append(graph_features(aggregate, G_seq30))

            g_sq_nodes = []
            g_sq_edges = []
            g_sq_degree = []
            g_sq_density = []
            g_sq_diameter = []
            g_sq_aspl = []
            g_sq_lscc = []
            g_sq_pe = []
            g_sq_l1 = []
            g_sq_l2 = []
            g_sq_l3 = []
            g_sq_cc = []
            g_sq_largestclique = []
            g_sq_zlscc = []
            g_sq_zdiameter = []
            g_sq_zaspl = []
            g_sq_zcc = []
            for each_entry in dynamic_entry:
                g_sq_nodes.append(each_entry["g_sq_nodes"])
                g_sq_edges.append(each_entry["g_sq_edges"])
                g_sq_degree.append(each_entry["g_sq_degree"])
                g_sq_density.append(each_entry["g_sq_density"])
                g_sq_diameter.append(each_entry["g_sq_diameter"])
                g_sq_aspl.append(each_entry["g_sq_aspl"])
                g_sq_lscc.append(each_entry["g_sq_lscc"])
                g_sq_pe.append(each_entry["g_sq_pe"])
                g_sq_l1.append(each_entry["g_sq_l1"])
                g_sq_l2.append(each_entry["g_sq_l2"])
                g_sq_l3.append(each_entry["g_sq_l3"])
                g_sq_cc.append(each_entry["g_sq_cc"])
                g_sq_largestclique.append(each_entry["g_sq_largestclique"])
                g_sq_zlscc.append(each_entry["g_sq_zlscc"])
                g_sq_zdiameter.append(each_entry["g_sq_zdiameter"])
                g_sq_zaspl.append(each_entry["g_sq_zaspl"])
                g_sq_zcc.append(each_entry["g_sq_zcc"])

            entry["g_30sq10_nodes"] = mean(g_sq_nodes)
            entry["g_30sq10_edges"] = mean(g_sq_edges)
            entry["g_30sq10_degree"] = mean(g_sq_degree)
            entry["g_30sq10_density"] = mean(g_sq_density)
            entry["g_30sq10_diameter"] = mean(g_sq_diameter)
            entry["g_30sq10_aspl"] = mean(g_sq_aspl)
            entry["g_30sq10_lscc"] = mean(g_sq_lscc)
            entry["g_30sq10_pe"] = mean(g_sq_pe)
            entry["g_30sq10_l1"] = mean(g_sq_l1)
            entry["g_30sq10_l2"] = mean(g_sq_l2)
            entry["g_30sq10_l3"] = mean(g_sq_l3)
            entry["g_30sq10_cc"] = mean(g_sq_cc)
            entry["g_30sq10_largestclique"] = mean(g_sq_largestclique)
            entry["g_30sq10_zlscc"] = mean(g_sq_zlscc)
            entry["g_30sq10_zdiameter"] = mean(g_sq_zdiameter)
            entry["g_30sq10_zaspl"] = mean(g_sq_zaspl)
            entry["g_30sq10_zcc"] = mean(g_sq_zcc)
        else:
            entry["g_30sq10_nodes"] = entry["g_sq_nodes"]
            entry["g_30sq10_edges"] = entry["g_sq_edges"]
            entry["g_30sq10_degree"] = entry["g_sq_degree"]
            entry["g_30sq10_density"] = entry["g_sq_density"]
            entry["g_30sq10_diameter"] = entry["g_sq_diameter"]
            entry["g_30sq10_aspl"] = entry["g_sq_aspl"]
            entry["g_30sq10_lscc"] = entry["g_sq_lscc"]
            entry["g_30sq10_pe"] = entry["g_sq_pe"]
            entry["g_30sq10_l1"] = entry["g_sq_l1"]
            entry["g_30sq10_l2"] = entry["g_sq_l2"]
            entry["g_30sq10_l3"] = entry["g_sq_l3"]
            entry["g_30sq10_cc"] = entry["g_sq_cc"]
            entry["g_30sq10_largestclique"] = entry["g_sq_largestclique"]
            entry["g_30sq10_zlscc"] = entry["g_sq_zlscc"]
            entry["g_30sq10_zdiameter"] = entry["g_sq_zdiameter"]
            entry["g_30sq10_zaspl"] = entry["g_sq_zaspl"]
            entry["g_30sq10_zcc"] = entry["g_sq_zcc"]

        df_strgfeatures = pd.concat(
            [df_strgfeatures, pd.DataFrame([entry])], ignore_index=True
        )

df_strgfeatures.to_csv(
    root_data + "Remora-2023//Batch-1//features//6b1_sequential_graph_features.csv"
)


# In[ ]:


len(node_sequence)


# In[8]:


df_strgfeatures


# In[120]:


dynamic_entry


# In[84]:


entry["g_sq_zlscc"]


# In[50]:


l2 = 0
l3 = 0
for node1 in nx.nodes(G_seq):
    # print(node1)
    for node2 in nx.neighbors(G_seq, node1):
        # print('\t', node2)
        if node1 in nx.neighbors(G_seq, node2):
            # print('diad')
            l2 += 1
        for node3 in nx.neighbors(G_seq, node2):
            if node1 in nx.neighbors(G_seq, node3):
                if node1 != node2 and node2 != node3:
                    l3 += 1
l3


# In[23]:


sorted(nx.simple_cycles(G_seq, length_bound=3))


# In[12]:


sorted(nx.simple_cycles(G_seq, length_bound=1))


# In[30]:


for each_item in nx.selfloop_edges(G_seq):
    print(each_item)


# In[ ]:
