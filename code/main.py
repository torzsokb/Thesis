import gurobipy as gp
from gurobipy import GRB
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random



def read_data(path, rand_weight=False):
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
                graph.add_edge(d_id, r_id, weight=weight)


    return graph



def is_chain_position_valid(i, j, k, L, d=2, ndd=False):

    if ndd:
        return k == 1
    else:
        return d <= k and k <= L

def draw_solution(graph, cycles, x_vals, y_vals):

    selected_edges = []
    selected_cycles = [cycle for cycle, value in x_vals.items() if value >= 0.5]

    for cycle in selected_cycles:
        selected_nodes = cycles[cycle]
        last = len(selected_nodes) - 1
        for i in range(last):
            selected_edges.append((selected_nodes[i], selected_nodes[i+1]))
        selected_edges.append((selected_nodes[last], selected_nodes[0]))

    selected_arcs = [k for k, v in y_vals.items() if v >= 0.5]
        
    for edge in selected_arcs:
        selected_edges.append((edge[0], edge[1]))
        
        
    edge_colors = ['black' if edge in selected_edges else 'lightgrey' for edge in graph.edges()]
    pos = nx.spring_layout(graph, seed=23, k=2)
    
    nx.draw(graph, pos, with_labels=True, edge_color=edge_colors, node_color='skyblue',
        node_size=150, arrows=True, connectionstyle='arc3,rad=0.1')
    plt.show()

def add_y_vars(model, y, graph, edge, L, p):

    w = graph.edges[edge]["weight"]
    i = edge[0]
    j = edge[1]
    if graph.nodes[i]["ndd"]:
        y[i, j, 1] = model.addVar(vtype=GRB.BINARY, obj=w*p, name=f"y[{i},{j},{1}]")
    else:
        for k in range(2, L + 1):
            y[i, j, k] = model.addVar(vtype=GRB.BINARY, obj=w*p**k, name=f"y[{i},{j},{k}]")
    
def add_x_cycle_var(model, x, graph, cycle, c, p):
    k = len(cycle)
    w = 0
    for i in range(k - 1):
        w += graph[cycle[i]][cycle[i+1]]["weight"]
    w += graph[cycle[k - 1]][cycle[0]]["weight"]

    x[c] = model.addVar(vtype=GRB.BINARY, obj=w*(p**k), name=f"x[{c}]")

def add_ndd_cap_constr(model, graph, y, node):
    expr = gp.LinExpr()
    for successor in graph.successors(node):
        expr += y[node, successor, 1]
                
    model.addConstr(expr <= 1, name=f"4c_{node}")

def add_patient_cap_constr_cycle(model, graph, cycles, L, y, x, node):
    
    expr =  gp.LinExpr()
    for c, cycle in enumerate(cycles):
        if node in cycle:
            expr += x[c]
    if L > 0:
        for predecessor in graph.predecessors(node):
            if graph.nodes[predecessor]["ndd"]:
                expr += y[predecessor, node, 1]
            else:
                for k in range(2, L + 1):
                    expr += y[predecessor, node, k]
    model.addConstr(expr <= 1, name=f"4b_{node}")

def add_flow_cons_constr(model, graph, L, y, node):
    for k in range(1, L):
        expr = gp.LinExpr()
        for predecessor in graph.predecessors(node):
            if graph.nodes[predecessor]["ndd"]:
                if k == 1:
                    expr += y[predecessor, node, 1]
            else:
                if k > 1:
                    expr += y[predecessor, node, k]
        for successor in graph.successors(node):
            expr -= y[node, successor, k+1]
        model.addConstr(expr >= 0, name=f"4d_{node}_{k}")

def match_kidneys(graph: nx.DiGraph, K=3, L=4, version="PICEF", p=1, consider_leftover=False):

    versions = ["PICEF", "HPICEF", "PIEF"]
    assert version in versions, "version invalid"

    if version == "PIEF":
        L = 0

    model = gp.Model()
    model.ModelSense = GRB.MAXIMIZE
    
    y = {}
    x = {}

    if L > 0:
        for edge in graph.edges:
            add_y_vars(model=model, graph=graph, y=y, edge=edge, L=L, p=p)
    
    if version == "PICEF":

        cycles = list(nx.simple_cycles(graph, length_bound=K))

        for c, cycle in enumerate(cycles):
            add_x_cycle_var(model=model, x=x, cycle=cycle, c=c, p=p, graph=graph)
        
        for node in graph.nodes:

            if graph.nodes[node]["ndd"] and L > 0:
                add_ndd_cap_constr(model=model, graph=graph, y=y, node=node)
            else:
                # print("not ndd")
                add_patient_cap_constr_cycle(model=model, graph=graph, cycles=cycles, L=L, x=x, y=y, node=node)
                add_flow_cons_constr(model=model, graph=graph, L=L, y=y, node=node)

        model.optimize()

        y_vals = {i: y[i].X for i in y}
        x_vals = {i: x[i].X for i in x}

        draw_solution(graph=graph, cycles=cycles, x_vals=x_vals, y_vals=y_vals)


    else:
        ...       

    
    


def test():

    Z = [[0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]]
    
    for i in range(len(Z)):
        if Z[i][i] == 1:
            Z[i][i] = 0
    

    comp_matrix = np.asarray(Z)
    print(comp_matrix.sum())
    dg = nx.from_numpy_array(comp_matrix, create_using=nx.DiGraph)

    for node in dg.nodes:
        dg.nodes[node]["ndd"] = dg.in_degree(node) == 0
        dg.nodes[node]["leftover_value"] = 0.5

    for edge in dg.edges:
        dg.edges[edge]["weight"] = 1

    for node in dg.nodes(data=True): 
        print(node)
    match_kidneys(dg)


def main():
    path = "/Users/barnitorzsok/Documents/GitHub/Thesis/data/Instances/InstancesM/outputM-1.xlsx"
    graph = read_data(path, rand_weight=True)
    ndds = [n for n in graph.nodes if graph.nodes[n]["ndd"]]
    print(f"n: {len(graph.nodes)}, ndds: {len(ndds)}, arcs: {len(graph.edges)}")
    
    random.seed(23)

    

    fraction_to_remove = 0.8
    num_edges_to_remove = int(fraction_to_remove * graph.number_of_edges())
    edges_to_remove = random.sample(list(graph.edges), num_edges_to_remove)
    graph.remove_edges_from(edges_to_remove)

    fraction_to_remove = 0.7
    num_ndds_to_remove = int(fraction_to_remove * len(ndds))
    ndds_to_remove = random.sample(ndds, num_ndds_to_remove)
    graph.remove_nodes_from(ndds_to_remove)

    


    match_kidneys(graph=graph, K=3, L=4, p=1)



if __name__ == "__main__":
    main()