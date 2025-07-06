import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random
import xml.etree.ElementTree as ET
import json
import pickle
import time

def load_graph_from_xml(path: str, use_weights=False) -> nx.DiGraph:

    tree = ET.parse(path)
    root = tree.getroot()

    recipients = {}
    for r in root.find("recipients"):
        rid = int(r.get("recip_id"))
        recipients[rid] = {
            "cPRA": float(r.get("cPRA")),
            "patient_bloodtype": r.get("bloodtype"),
            "has_blood_comp_donor": r.get("hasBloodCompatibleDonor") == "true"
        }

    G = nx.DiGraph()

    for entry in root.findall("entry"):

        node_id = int(entry.get("donor_id"))
        donor_bloodtype = entry.get("bloodtype")
        donor_age = int(entry.findtext("dage"))

        altruistic = entry.findtext("altruistic") == "true"

        if altruistic:
            G.add_node(node_id, donor_bloodtype=donor_bloodtype, donor_age=donor_age, ndd=True)
        else:
            src_elem = entry.find("sources/source")
            if src_elem is None:
                raise ValueError(f"Entry {node_id} has no <sources> or <altruistic> tag")
            
            attrs = {
                "donor_bloodtype": donor_bloodtype,
                "donor_age": donor_age,
                **recipients.get(node_id, {}),
                "ndd": False
            }
            G.add_node(node_id, **attrs)

    
    for entry in root.findall("entry"):
        node_id = int(entry.get("donor_id"))
        for m in entry.findall("matches/match"):
            recipient_node_id  = int(m.findtext("recipient"))
            if use_weights:
                # score = float(m.findtext("score"))
                score = 0.1 + (float(m.findtext("score") - 1) / 100)
                G.add_edge(node_id, recipient_node_id, score=score)
            else:
                G.add_edge(node_id, recipient_node_id)

    # print("Total nodes:", G.number_of_nodes())
    # print("Total edges:", G.number_of_edges())

    # altr = [n for n,d in G.nodes(data=True) if d["ndd"]]
    # print("Altruists:", altr)

    return G

def load_graph_from_json(path_dir: str, use_weights=False, use_full_details=False, instance_id: str=0) -> nx.DiGraph:

    config_data = get_config_data(path=f"{path_dir}/config.json")
    if use_full_details:
        assert config_data["fullDetails"] == True, "file does not have full details"

    instance_path = f"{path_dir}/genjson-{instance_id}.json"
    with open(instance_path, 'r') as f:
            raw = json.load(f)

    graph = nx.DiGraph(**config_data)

    recipients = raw["recipients"]
    donors = list(raw["data"].keys())
    print(type(recipients))

    for donor_id in donors:
        
        node_id = int(donor_id)
        entry = raw["data"][donor_id]
        
        sources = entry.get("sources", [])
        ndd = (len(sources) == 0)
        attrs = {"ndd": ndd}
        if use_full_details:
            attrs["donor_bloodtype"] = entry["bloodtype"]
            attrs["donor_age"] = entry["dage"]
            if not ndd:
                recipient = recipients[str(sources[0])]
                attrs["cPRA"] = recipient["cPRA"]
                attrs["recipient_blood_type"] = recipient["bloodtype"]
                attrs["has_bloodtype_comp_donor"] = recipient["hasBloodCompatibleDonor"]

        graph.add_node(node_id, **attrs)
    
    for donor_id in donors:

        node_id = int(donor_id)
        entry = raw["data"][donor_id]
        matches = entry.get("matches", [])
        for m in matches:
            rec_id = int(m["recipient"])
            if use_weights:
                score =  0.1 + (float(m["score"]) - 1) / 100
                graph.add_edge(node_id, rec_id, score=score)
            else:
                graph.add_edge(node_id, rec_id)

    return graph

def get_config_data(path: str) -> dict:
        with open(path, 'r') as f:
            raw = json.load(f)
            return raw

def read_data(path: str, rand_weight=False) -> nx.DiGraph:
    random.seed(23)
    excel_file = pd.ExcelFile(path)

    compatibilities = pd.read_excel(excel_file, "Compatibilities", index_col=0)
    compatibilities = compatibilities.fillna(0)
    print(compatibilities.head())

    donors = pd.read_excel(excel_file, "Donors")
    print(donors.head())
    d_ids = list(np.unique(donors["ID"]))

    recipients = pd.read_excel(excel_file, "Recipients")
    print(recipients.head())
    r_ids = list(np.unique(recipients["ID"]))

    graph = nx.DiGraph()

    for _, recipient in recipients.iterrows():
        id = recipient["ID"]
        blood_type = recipient["Blood Type"]
        cPRA = recipient["cPRA"]
        graph.add_node(id, node_id=id, rec_blood_type=blood_type, cPRA=cPRA, ndd=False)

    for _, donor in donors.iterrows():
        id = donor["ID"]
        blood_type = donor["Blood Type"]
        age = donor["Age"]
        if id in r_ids:
            graph.nodes[id]["donor_blood_type"] = blood_type
            graph.nodes[id]["donor_age"] = age
        else:
            graph.add_node(id, node_id=id, donor_blood_type=blood_type, donor_age=age, ndd=True)
    
    for d_id in d_ids:
        for r_id in r_ids:
            if compatibilities.loc[d_id][str(r_id)] == 1.0:
                weight = 1
                if rand_weight:
                    weight -= random.random() * 0.5
                graph.add_edge(d_id, r_id, score=weight)


    return graph

def save_converted_graph(graph: nx.DiGraph, out_path: str) -> None:
    with open(out_path, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_graph_from_pickle(path: str) -> nx.DiGraph:
    with open(path, "rb") as f:
        nx_graph = pickle.load(f)
        print("data opened!")
    return nx_graph

def main():
   
    # start = time.time()
    # path = "data/simulation/converted/simulation_3.pkl"
    # graph = load_graph_from_pickle(path=path)
    # print(graph.graph)
    # end = time.time()
    # print(f"loading from pickle took {end-start}")
    # g = graph.subgraph(random.sample(list(graph.nodes), 150))
    # print(g.graph)
    n = 0


    start = time.time()
    path_dir = f"data/generated_4"
    graph = load_graph_from_json(path_dir=path_dir, use_full_details=True, use_weights=True)
    save_converted_graph(graph=graph, out_path=f"{path_dir}/converted_{n}.pkl")
    print(graph.graph)
    end = time.time()
    print(f"loading from json took {end-start}")
    ...

if __name__ == "__main__":
    main()