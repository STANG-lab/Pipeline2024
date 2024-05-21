#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import ast
import os
import torch
import re
import networkx as nx
from statistics import mean
from statistics import stdev


# In[2]:


import spacy

nlp = spacy.load("en_core_web_sm")


# In[3]:


### Points to the folder that contains all raw
root_data = "C://Users//ANikzad//Desktop//Local_Pipeline//Data//"

### Specify Source Folder
# --- Remora
dicourse_feature_location_semgraph = (
    root_data
    + "Remora-2023//Batch-1//6_discourse_level_features//6b_graphs//6b2_semgraphs//raw_df//"
)

### Specify Destination folder
# --- Remora
dicourse_feature_location_semgraphfeatures = (
    root_data
    + "Remora-2023//Batch-1//6_discourse_level_features//6b_graphs//6b2_semgraphs//refined_df//"
)

dicourse_feature_location_peripheryedges = (
    root_data
    + "Remora-2023//Batch-1//6_discourse_level_features//6b_graphs//6b2_semgraphs//periphery_edgelist//"
)

dicourse_feature_location_coreedges = (
    root_data
    + "Remora-2023//Batch-1//6_discourse_level_features//6b_graphs//6b2_semgraphs//core_edgelist//"
)


# In[4]:


# get all the files in the transcripts folder (the folder should just include transcripts)
all_aggregate_files = os.listdir(dicourse_feature_location_semgraph)
len(all_aggregate_files)


# In[5]:


# Define a function to convert a string representation of a list to a list
def convert_to_list(string_repr):
    try:
        return ast.literal_eval(string_repr)
    except (SyntaxError, ValueError):
        # Handle any potential errors when parsing the string
        return []


def coref_to_list(coref_str):
    try:
        return coref_str.split(", ")
    except:
        return []


def head_extract(text):
    doc = nlp(text)
    # Iterate over the words in the parsed text
    for token in doc:
        # Check if the token is a head (root) of a subtree
        if token.dep_ == "ROOT":
            head = token
            break
    return str(head)


def core_extract(text):
    doc = nlp(text)
    output_list = []
    for noun_chunk in doc.noun_chunks:
        output_list.append(head_extract(noun_chunk.text))
    for token in doc:
        if token.dep_ == "ROOT":
            if str(token) not in output_list:
                output_list.append(str(token))

    return " ".join(output_list)


def extract_lemma(text):
    doc = nlp(text)
    lemmas = []
    for token in doc:
        if token.tag_ not in ["IN", "WP"]:
            if str(token.lemma_) not in lemmas:
                lemmas.append(str(token.lemma_))
    if text in ["like", "because"]:
        lemmas.append(text)
    return " ".join(lemmas)


# In[6]:


arguments = [
    "ARG0",
    "C-ARG0",
    "R-ARG0",
    "ARG1",
    "R-ARG1",
    "C-ARG1",
    "ARG2",
    "R-ARG2",
    "C-ARG2",
    "ARG3",
    "ARG4",
    "ARG5",
]
predicates = [
    "V",
]
manner = [
    "ARGM-ADV",
    "ARGM-MNR",
    "C-ARGM-MNR",
    "R-ARGM-MNR",
    "ARGM-PRD",
    "ARGM-EXT",
    "ARGM-COM",
    "ARGM-ADJ",
]
setting = [
    "ARGM-TMP",
    "ARGM-LOC",
    "R-ARGM-LOC",
    "R-ARGM-TMP",
    "ARGM-DIR",
    "ARGM-GOL",
]
cause = [
    "ARGM-CAU",
    "ARGM-PRP",
    "ARGM-PNC",
]
all_roles = arguments + predicates + manner + setting + cause


# In[68]:


### Graph Feature Function ###


def graph_feature(aggregate, G, g_type):
    entry = {}
    entry["uid"] = aggregate.split(".")[0]
    entry[f"g_{g_type}_nodes"] = G.number_of_nodes()
    entry[f"g_{g_type}_edges"] = G.number_of_edges()

    try:
        entry[f"g_{g_type}_degree"] = sum([d for (n, d) in nx.degree(G)]) / float(
            G.number_of_nodes()
        )
    except:
        entry[f"g_{g_type}_degree"] = 0
    try:
        entry[f"g_{g_type}_density"] = nx.density(G)
    except:
        entry[f"g_{g_type}_density"] = 0
    try:
        entry[f"g_{g_type}_diameter"] = nx.diameter(nx.to_undirected(G))
    except:
        pass
    try:
        entry[f"g_{g_type}_aspl"] = nx.average_shortest_path_length(nx.to_undirected(G))
    except:
        pass
    try:
        entry[f"g_{g_type}_lscc"] = len(
            max(nx.strongly_connected_components(G), key=len)
        )
    except:
        entry[f"g_{g_type}_lscc"] = 0
    try:
        entry[f"g_{g_type}_ncc"] = len(
            sorted(nx.connected_components(nx.Graph(G)), key=len, reverse=True)
        )
    except:
        entry[f"g_{g_type}_ncc"] = 0
    try:
        entry[f"g_{g_type}_lcc"] = len(
            max(nx.connected_components(nx.Graph(G)), key=len)
        )
    except:
        entry[f"g_{g_type}_lcc"] = 0

    parallel_edge = 0
    l2 = 0
    l3 = 0
    for node1 in nx.nodes(G):
        # print(node1)
        for node2 in nx.neighbors(G, node1):
            # print('\t', node2)
            parallel_edge = parallel_edge + (G.number_of_edges(u=node1, v=node2) - 1)
            if node1 in nx.neighbors(G, node2):
                l2 += 1
            for node3 in nx.neighbors(G, node2):
                if node1 in nx.neighbors(G, node3):
                    if node1 != node2 and node2 != node3:
                        l3 += 1
    entry[f"g_{g_type}_pe"] = parallel_edge
    entry[f"g_{g_type}_l1"] = nx.number_of_selfloops(G)
    entry[f"g_{g_type}_l2"] = l2 / 2
    entry[f"g_{g_type}_l3"] = l3 / 3

    G2 = nx.Graph()
    for u, v, data in G.edges(data=True):
        w = data["weight"] if "weight" in data else 1.0
        if G2.has_edge(u, v):
            G2[u][v]["weight"] += w
        else:
            G2.add_edge(u, v, weight=w)
    try:
        entry[f"g_{g_type}_cc"] = nx.average_clustering(G2.to_directed())
    except:
        entry[f"g_{g_type}_cc"] = 0
    try:
        entry[f"g_{g_type}_largestclique"] = nx.approximation.large_clique_size(G2)
    except:
        entry[f"g_{g_type}_largestclique"] = 0

    diameter_rand = []
    aspl_rand = []
    lscc_rand = []
    lcc_rand = []
    ncc_rand = []
    cc_rand = []

    for seed in range(0, 200):
        G_rand = nx.gnm_random_graph(
            entry[f"g_{g_type}_nodes"],
            entry[f"g_{g_type}_edges"],
            seed=seed,
            directed=True,
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
            lcc_rand.append(
                len(max(nx.connected_components(nx.Graph(G_rand)), key=len))
            )
        except:
            pass
        try:
            ncc_rand.append(
                len(
                    sorted(
                        nx.connected_components(nx.Graph(G_rand)), key=len, reverse=True
                    )
                )
            )
        except:
            pass
    try:
        entry[f"g_{g_type}_zlscc"] = (
            entry[f"g_{g_type}_lscc"] - mean(lscc_rand)
        ) / stdev(lscc_rand)
    except:
        pass
    try:
        entry[f"g_{g_type}_zncc"] = (entry[f"g_{g_type}_ncc"] - mean(ncc_rand)) / stdev(
            ncc_rand
        )
    except:
        pass
    # try:
    entry[f"g_{g_type}_zlcc"] = (entry[f"g_{g_type}_lcc"] - mean(lcc_rand)) / stdev(
        lcc_rand
    )
    # except:
    #   entry[f'g_{g_type}_zlcc'] = np.nan
    try:
        entry[f"g_{g_type}_zdiameter"] = (
            entry[f"g_{g_type}_diameter"] - mean(diameter_rand)
        ) / stdev(diameter_rand)
    except:
        pass
    try:
        entry[f"g_{g_type}_zaspl"] = (
            entry[f"g_{g_type}_aspl"] - mean(aspl_rand)
        ) / stdev(aspl_rand)
    except:
        pass
    try:
        entry[f"g_{g_type}_zcc"] = (entry[f"g_{g_type}_cc"] - mean(cc_rand)) / stdev(
            cc_rand
        )
    except:
        pass
    return entry


# In[71]:


df_semgfeatures = pd.DataFrame()
for aggregate in all_aggregate_files[0:2]:
    if aggregate.split("_")[5] in ["JOU", "PIC"]:
        print(aggregate)
        df = pd.read_excel(dicourse_feature_location_semgraph + aggregate)
        df = df.loc[df["speaker"] == "Participant"].reset_index().copy()
        df["all_roles"] = df["all_roles"].apply(convert_to_list)
        df["clusters"] = df["clusters"].apply(coref_to_list)
        role_df = pd.DataFrame()

        df = df[
            [
                "utterance_index",
                "dialogue_tag",
                "token_index",
                "token_srl",
                "all_roles",
                "n_clusters",
                "clusters",
            ]
        ]
        tokens_dic = {}
        clusters = []
        for index, row in df.iterrows():
            tokens_dic[row["token_index"]] = row["token_srl"]
            for each_coref in row["clusters"]:
                if each_coref not in clusters:
                    clusters.append(each_coref)
        clusters_dic = {}

        for each_cluster in clusters:
            tokens = []
            for index, row in df.iterrows():
                if each_cluster in row["clusters"]:
                    tokens.append((row["token_index"], row["token_srl"]))
            merged_tokens = []
            token_numbers = []
            tokens.append(("null", "null"))
            for j in range(0, len(tokens) - 1):
                token_numbers.append(tokens[j][0])
                if tokens[j][0] + 1 != tokens[j + 1][0]:
                    merged_tokens.append((token_numbers[0], token_numbers[-1]))
                    token_numbers = []
            coref_head_list = []
            for each_item in merged_tokens:
                each_coref = []
                for k in range(each_item[0], each_item[1] + 1):
                    each_coref.append(tokens_dic[k])
                each_coref = " ".join(each_coref)
                coref_head_list.append(core_extract(each_coref))

            cluster_head = "*TBD*"
            for each_item in coref_head_list:
                if len(each_item.split(" ")) == 1:
                    doc = nlp(each_item)

                    if doc[0].tag_ in ["NN", "NNS", "NNP"]:
                        cluster_head = each_item
                        break
            if cluster_head == "*TBD*":
                for each_item in coref_head_list:
                    if len(each_item.split(" ")) == 1:
                        doc = nlp(each_item)

                        if doc[0].tag_ in ["PRP", "PRP%"]:
                            cluster_head = each_item
                            break
            if cluster_head == "*TBD*":
                head_list = []
                for l in range(merged_tokens[0][0], merged_tokens[0][1] + 1):
                    head_list.append(tokens_dic[l])
                cluster_head = " ".join(head_list)

            for each_item in merged_tokens:
                clusters_dic[each_item] = cluster_head

        for index, row in df.iterrows():
            n_roles = len(row["all_roles"])
            df.at[index, "n_roles"] = n_roles
        for n, sub_df in df.groupby("utterance_index"):
            n_roles = int(sub_df["n_roles"].unique()[0])

            for i in range(0, n_roles):
                entry = {}
                entry["utterance_index"] = n
                utterance_roles = []
                for index, row in sub_df.iterrows():
                    if row["all_roles"][i] not in utterance_roles:
                        utterance_roles.append(row["all_roles"][i])
                for each_role in utterance_roles:
                    if each_role in all_roles:
                        role_tokens = []
                        role_tokens_content = []
                        for index, row in sub_df.iterrows():
                            if row["all_roles"][i] == each_role:
                                role_tokens.append(row["token_index"])
                        try:
                            role_tokens_content.append(
                                clusters_dic[(role_tokens[0], role_tokens[-1])]
                            )

                        except:
                            roles_address = (role_tokens[0], role_tokens[-1])
                            for j in range(roles_address[0], roles_address[1] + 1):
                                try:
                                    role_tokens_content.append(tokens_dic[j])
                                except:
                                    pass

                        content_list = []

                        entry[each_role] = extract_lemma(
                            core_extract(
                                " ".join(role_tokens_content)
                                .replace(" ' ", "'")
                                .replace(" '", "'")
                                .replace(" ,", ",")
                            )
                        )
                if len(entry) > 2:
                    role_df = pd.concat(
                        [role_df, pd.DataFrame([entry])], ignore_index=1
                    )

        role_df.to_csv(
            dicourse_feature_location_semgraphfeatures
            + aggregate.split(".")[0]
            + ".csv"
        )

        sentence_list = role_df["utterance_index"].unique().tolist()

        G_core = nx.MultiDiGraph()
        G_periphery = nx.MultiDiGraph()

        for index, row in role_df.iterrows():
            sentence_id = row["utterance_index"]
            if row["V"] not in ["'", ""] and type(row["V"]) == str:
                # print(row['V'], type(row['V']))
                for each_argument in arguments:
                    try:
                        if (
                            type(row[each_argument]) == str
                            and row[each_argument] != " "
                        ):
                            for element in row[each_argument].split(" "):
                                G_core.add_edge(
                                    row["V"],
                                    element,
                                    weight=1,
                                    category=each_argument,
                                    sentence=sentence_id,
                                )
                    except:
                        pass

                if "ARG0" in role_df.columns:
                    if type(row["ARG0"]) == str:
                        if "ARG1" in role_df.columns:
                            if type(row["ARG1"]) == str:
                                for each_actor in row["ARG0"].split(" "):
                                    for each_patient in row["ARG1"].split(" "):
                                        G_core.add_edge(
                                            each_actor,
                                            each_patient,
                                            weight=1,
                                            category="action",
                                            sentence=sentence_id,
                                        )
                        if "ARG2" in role_df.columns:
                            if type(row["ARG2"]) == str:
                                for each_actor in row["ARG0"].split(" "):
                                    for each_patient in row["ARG2"].split(" "):
                                        G_core.add_edge(
                                            each_actor,
                                            each_patient,
                                            weight=1,
                                            category="action",
                                            sentence=sentence_id,
                                        )

                for each_adjunct in setting + cause + manner:
                    try:
                        if type(row[each_adjunct]) == str:
                            for element in row[each_adjunct].split(" "):
                                G_periphery.add_edge(
                                    element,
                                    row["V"],
                                    weight=1,
                                    category=each_adjunct,
                                    sentence=sentence_id,
                                )
                    except:
                        pass

        nx.write_edgelist(
            G_core,
            dicourse_feature_location_coreedges + aggregate.split(".")[0] + ".txt",
            data=["weight", "category"],
        )
        nx.write_edgelist(
            G_periphery,
            dicourse_feature_location_peripheryedges + aggregate.split(".")[0] + ".txt",
            data=["weight", "category"],
        )

        entry_cr = graph_feature(aggregate, G_core, "cr")
        entry_pr = graph_feature(aggregate, G_periphery, "pr")

        entry = entry_cr | entry_pr

        m = 0
        if len(sentence_list) > 4:
            dynamic_entry_cr = []
            dynamic_entry_pr = []
            while m + 2 < len(sentence_list):
                sentence_window = sentence_list[m : m + 3]

                G_core_dynamic = nx.MultiDiGraph()
                G_periphery_dynamic = nx.MultiDiGraph()
                for u, v, data in G_core.edges(data=True):
                    if data["sentence"] in sentence_window:
                        G_core_dynamic.add_edge(u, v)
                for u, v, data in G_periphery.edges(data=True):
                    if data["sentence"] in sentence_window:
                        G_periphery_dynamic.add_edge(u, v)
                dynamic_entry_cr.append(
                    graph_feature(aggregate, G_core_dynamic, "3cr1")
                )
                dynamic_entry_pr.append(
                    graph_feature(aggregate, G_periphery_dynamic, "3pr1")
                )

                m += 1

            g_cr_nodes = []
            g_cr_edges = []
            g_cr_degree = []
            g_cr_density = []
            g_cr_diameter = []
            g_cr_aspl = []
            g_cr_lscc = []
            g_cr_lcc = []
            g_cr_ncc = []
            g_cr_pe = []
            g_cr_l1 = []
            g_cr_l2 = []
            g_cr_l3 = []
            g_cr_cc = []
            g_cr_largestclique = []
            g_cr_zlscc = []
            g_cr_zlcc = []
            g_cr_zncc = []
            g_cr_zdiameter = []
            g_cr_zaspl = []
            g_cr_zcc = []
            for each_entry in dynamic_entry_cr:
                g_cr_nodes.append(each_entry["g_3cr1_nodes"])
                g_cr_edges.append(each_entry["g_3cr1_edges"])
                g_cr_degree.append(each_entry["g_3cr1_degree"])
                g_cr_density.append(each_entry["g_3cr1_density"])
                g_cr_diameter.append(each_entry["g_3cr1_diameter"])
                g_cr_aspl.append(each_entry["g_3cr1_aspl"])
                g_cr_lscc.append(each_entry["g_3cr1_lscc"])
                g_cr_lcc.append(each_entry["g_3cr1_lcc"])
                g_cr_ncc.append(each_entry["g_3cr1_ncc"])
                g_cr_pe.append(each_entry["g_3cr1_pe"])
                g_cr_l1.append(each_entry["g_3cr1_l1"])
                g_cr_l2.append(each_entry["g_3cr1_l2"])
                g_cr_l3.append(each_entry["g_3cr1_l3"])
                g_cr_cc.append(each_entry["g_3cr1_cc"])
                g_cr_largestclique.append(each_entry["g_3cr1_largestclique"])
                g_cr_zlscc.append(each_entry["g_3cr1_zlscc"])
                g_cr_zlcc.append(each_entry["g_3cr1_zlcc"])
                g_cr_zncc.append(each_entry["g_3cr1_zncc"])
                g_cr_zdiameter.append(each_entry["g_3cr1_zdiameter"])
                g_cr_zaspl.append(each_entry["g_3cr1_zaspl"])
                g_cr_zcc.append(each_entry["g_3cr1_zcc"])

            entry["g_3cr1_nodes"] = mean(g_cr_nodes)
            entry["g_3cr1_edges"] = mean(g_cr_edges)
            entry["g_3cr1_degree"] = mean(g_cr_degree)
            entry["g_3cr1_density"] = mean(g_cr_density)
            entry["g_3cr1_diameter"] = mean(g_cr_diameter)
            entry["g_3cr1_aspl"] = mean(g_cr_aspl)
            entry["g_3cr1_lscc"] = mean(g_cr_lscc)
            entry["g_3cr1_lcc"] = mean(g_cr_lcc)
            entry["g_3cr1_ncc"] = mean(g_cr_ncc)
            entry["g_3cr1_pe"] = mean(g_cr_pe)
            entry["g_3cr1_l1"] = mean(g_cr_l1)
            entry["g_3cr1_l2"] = mean(g_cr_l2)
            entry["g_3cr1_l3"] = mean(g_cr_l3)
            entry["g_3cr1_cc"] = mean(g_cr_cc)
            entry["g_3cr1_largestclique"] = mean(g_cr_largestclique)
            entry["g_3cr1_zlscc"] = mean(g_cr_zlscc)
            entry["g_3cr1_zlcc"] = mean(g_cr_zlcc)
            entry["g_3cr1_zncc"] = mean(g_cr_zncc)
            entry["g_3cr1_zdiameter"] = mean(g_cr_zdiameter)
            entry["g_3cr1_zaspl"] = mean(g_cr_zaspl)
            entry["g_3cr1_zcc"] = mean(g_cr_zcc)

            g_pr_nodes = []
            g_pr_edges = []
            g_pr_degree = []
            g_pr_density = []
            g_pr_diameter = []
            g_pr_aspl = []
            g_pr_lscc = []
            g_pr_lcc = []
            g_pr_ncc = []
            g_pr_pe = []
            g_pr_l1 = []
            g_pr_l2 = []
            g_pr_l3 = []
            g_pr_cc = []
            g_pr_largestclique = []
            g_pr_zlscc = []
            g_pr_zlcc = []
            g_pr_zncc = []
            g_pr_zdiameter = []
            g_pr_zaspl = []
            g_pr_zcc = []
            for each_entry in dynamic_entry_pr:
                g_pr_nodes.append(each_entry["g_3pr1_nodes"])
                g_pr_edges.append(each_entry["g_3pr1_edges"])
                g_pr_degree.append(each_entry["g_3pr1_degree"])
                g_pr_density.append(each_entry["g_3pr1_density"])
                g_pr_diameter.append(each_entry["g_3pr1_diameter"])
                g_pr_aspl.append(each_entry["g_3pr1_aspl"])
                g_pr_lscc.append(each_entry["g_3pr1_lscc"])
                g_pr_lcc.append(each_entry["g_3pr1_lcc"])
                g_pr_ncc.append(each_entry["g_3pr1_ncc"])
                g_pr_pe.append(each_entry["g_3pr1_pe"])
                g_pr_l1.append(each_entry["g_3pr1_l1"])
                g_pr_l2.append(each_entry["g_3pr1_l2"])
                g_pr_l3.append(each_entry["g_3pr1_l3"])
                g_pr_cc.append(each_entry["g_3pr1_cc"])
                g_pr_largestclique.append(each_entry["g_3pr1_largestclique"])
                g_pr_zlscc.append(each_entry["g_3pr1_zlscc"])
                g_pr_zlcc.append(each_entry["g_3pr1_zlcc"])
                g_pr_zncc.append(each_entry["g_3pr1_zncc"])
                g_pr_zdiameter.append(each_entry["g_3pr1_zdiameter"])
                g_pr_zaspl.append(each_entry["g_3pr1_zaspl"])
                g_pr_zcc.append(each_entry["g_3pr1_zcc"])

            entry["g_3pr1_nodes"] = mean(g_pr_nodes)
            entry["g_3pr1_edges"] = mean(g_pr_edges)
            entry["g_3pr1_degree"] = mean(g_pr_degree)
            entry["g_3pr1_density"] = mean(g_pr_density)
            entry["g_3pr1_diameter"] = mean(g_pr_diameter)
            entry["g_3pr1_aspl"] = mean(g_pr_aspl)
            entry["g_3pr1_lscc"] = mean(g_pr_lscc)
            entry["g_3pr1_lcc"] = mean(g_pr_lcc)
            entry["g_3pr1_ncc"] = mean(g_pr_ncc)
            entry["g_3pr1_pe"] = mean(g_pr_pe)
            entry["g_3pr1_l1"] = mean(g_pr_l1)
            entry["g_3pr1_l2"] = mean(g_pr_l2)
            entry["g_3pr1_l3"] = mean(g_pr_l3)
            entry["g_3pr1_cc"] = mean(g_pr_cc)
            entry["g_3pr1_largestclique"] = mean(g_pr_largestclique)
            entry["g_3pr1_zlscc"] = mean(g_pr_zlscc)
            entry["g_3pr1_zlcc"] = mean(g_pr_zlcc)
            entry["g_3pr1_zncc"] = mean(g_pr_zncc)
            entry["g_3pr1_zdiameter"] = mean(g_pr_zdiameter)
            entry["g_3pr1_zaspl"] = mean(g_pr_zaspl)
            entry["g_3pr1_zcc"] = mean(g_pr_zcc)

        else:
            entry["g_3cr1_nodes"] = entry["g_cr_nodes"]
            entry["g_3cr1_edges"] = entry["g_cr_edges"]
            entry["g_3cr1_degree"] = entry["g_cr_degree"]
            entry["g_3cr1_density"] = entry["g_cr_density"]
            entry["g_3cr1_diameter"] = entry["g_cr_diameter"]
            entry["g_3cr1_aspl"] = entry["g_cr_aspl"]
            entry["g_3cr1_lscc"] = entry["g_cr_lscc"]
            entry["g_3cr1_lcc"] = entry["g_cr_lcc"]
            entry["g_3cr1_ncc"] = entry["g_cr_ncc"]
            entry["g_3cr1_pe"] = entry["g_cr_pe"]
            entry["g_3cr1_l1"] = entry["g_cr_l1"]
            entry["g_3cr1_l2"] = entry["g_cr_l2"]
            entry["g_3cr1_l3"] = entry["g_cr_l3"]
            entry["g_3cr1_cc"] = entry["g_cr_cc"]
            entry["g_3cr1_largestclique"] = entry["g_cr_largestclique"]
            entry["g_3cr1_zlscc"] = entry["g_cr_zlscc"]
            entry["g_3cr1_zlcc"] = entry["g_cr_zlcc"]
            entry["g_3cr1_zncc"] = entry["g_cr_zncc"]
            entry["g_3cr1_zdiameter"] = entry["g_cr_zdiameter"]
            entry["g_3cr1_zaspl"] = entry["g_cr_zaspl"]
            entry["g_3cr1_zcc"] = entry["g_cr_zcc"]

            entry["g_3pr1_nodes"] = entry["g_pr_nodes"]
            entry["g_3pr1_edges"] = entry["g_pr_edges"]
            entry["g_3pr1_degree"] = entry["g_pr_degree"]
            entry["g_3pr1_density"] = entry["g_pr_density"]
            entry["g_3pr1_diameter"] = entry["g_pr_diameter"]
            entry["g_3pr1_aspl"] = entry["g_pr_aspl"]
            entry["g_3pr1_lscc"] = entry["g_pr_lscc"]
            entry["g_3pr1_lcc"] = entry["g_pr_lcc"]
            entry["g_3pr1_ncc"] = entry["g_pr_ncc"]
            entry["g_3pr1_pe"] = entry["g_pr_pe"]
            entry["g_3pr1_l1"] = entry["g_pr_l1"]
            entry["g_3pr1_l2"] = entry["g_pr_l2"]
            entry["g_3pr1_l3"] = entry["g_pr_l3"]
            entry["g_3pr1_cc"] = entry["g_pr_cc"]
            entry["g_3pr1_largestclique"] = entry["g_pr_largestclique"]
            entry["g_3pr1_zlscc"] = entry["g_pr_zlscc"]
            entry["g_3pr1_zlcc"] = entry["g_pr_zlcc"]
            entry["g_3pr1_zncc"] = entry["g_pr_zncc"]
            entry["g_3pr1_zdiameter"] = entry["g_pr_zdiameter"]
            entry["g_3pr1_zaspl"] = entry["g_pr_zaspl"]
            entry["g_3pr1_zcc"] = entry["g_pr_zcc"]

        df_semgfeatures = pd.concat(
            [df_semgfeatures, pd.DataFrame([entry])], ignore_index=True
        )

# df_semgfeatures.to_csv(root_data+'Remora-2023//Batch-1//features//semantic_graph_features.csv')


# In[70]:


lcc_rand


# In[62]:


G_core_dynamic = nx.MultiDiGraph()
for u, v, data in G_core.edges(data=True):
    if data["sentence"] in [8, 9, 12]:
        G_core_dynamic.add_edge(u, v)
len(max(nx.connected_components(nx.Graph(G_core_dynamic)), key=len))


# In[58]:


# In[56]:


for each in dynamic_entry_cr:
    print(each)


# In[35]:


G_core_dynamic.edges(data=True)


# In[16]:


sentence_list


# In[51]:


entry


# In[50]:


# In[69]:


nx.write_edgelist(
    G_core, "C://Users//ANikzad//Desktop//test.txt", data=["weight", "category"]
)


# In[80]:


role_df


# In[81]:


df


# In[ ]:


entry = {}
entry["uid"] = aggregate.split(".")[0]
entry["core_nn"] = G_core.number_of_nodes()
entry["core_ne"] = G_core.number_of_edges()
entry["core_l1"] = nx.number_of_selfloops(G_core)
# entry['core_l2'] = G_core.number_of_nodes
# entry['core_l3'] = G_core.number_of_nodes
try:
    entry["core_ad"] = sum([d for (n, d) in nx.degree(G_core)]) / float(
        G_core.number_of_nodes()
    )
except:
    entry["core_ad"] = 0
entry["core_gd"] = nx.density(G_core)
try:
    entry["core_lscc"] = len(max(nx.strongly_connected_components(G_core), key=len))
except:
    entry["core_lscc"] = 0
try:
    entry["core_ncc"] = len(
        sorted(nx.connected_components(nx.Graph(G_core)), key=len, reverse=True)
    )
except:
    entry["core_ncc"] = 0
# entry['core_lq'] = G_core.number_of_nodes
G_core2 = nx.Graph()
for u, v, data in G_core.edges(data=True):
    w = data["weight"] if "weight" in data else 1.0
    if G_core2.has_edge(u, v):
        G_core2[u][v]["weight"] += w
    else:
        G_core2.add_edge(u, v, weight=w)
try:
    entry["core_cc"] = nx.average_clustering(G_core2.to_directed())
except:
    entry["core_cc"] = 0
entry["periphery_nn"] = G_periphery.number_of_nodes()
entry["periphery_ne"] = G_periphery.number_of_edges()
entry["periphery_l1"] = nx.number_of_selfloops(G_periphery)
# entry['periphery_l2'] = G_periphery.number_of_nodes
# entry['periphery_l3'] = G_periphery.number_of_nodes
try:
    entry["periphery_ad"] = sum([d for (n, d) in nx.degree(G_periphery)]) / float(
        G_periphery.number_of_nodes()
    )
except:
    entry["periphery_ad"] = 0
entry["periphery_gd"] = nx.density(G_periphery)
try:
    entry["periphery_lscc"] = len(
        max(nx.strongly_connected_components(G_periphery), key=len)
    )
except:
    entry["periphery_lscc"] = 0
try:
    entry["periphery_ncc"] = len(
        sorted(nx.connected_components(nx.Graph(G_periphery)), key=len, reverse=True)
    )
except:
    entry["periphery_ncc"] = 0
# entry['periphery_lq'] = G_periphery.number_of_nodes
G_periphery2 = nx.Graph()
for u, v, data in G_periphery.edges(data=True):
    w = data["weight"] if "weight" in data else 1.0
    if G_periphery2.has_edge(u, v):
        G_periphery2[u][v]["weight"] += w
    else:
        G_periphery2.add_edge(u, v, weight=w)
try:
    entry["periphery_cc"] = nx.average_clustering(G_periphery2.to_directed())
except:
    entry["periphery_cc"] = 0
df_semgfeatures = pd.concat([df_semgfeatures, pd.DataFrame([entry])], ignore_index=True)
df_semgfeatures.to_csv(
    root_data + "Remora-2023//Batch-1//features//semantic_graph_features.csv"
)
