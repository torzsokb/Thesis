import gurobipy as gp
from gurobipy import GRB
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

def draw_solution(graph: nx.DiGraph, cycles: list, x_vals: dict, y_vals: dict) -> None:

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

def add_y_vars(model: gp.Model, y: dict, graph: nx.DiGraph, edge, L: int, p: float, use_weights=False) -> None:
    w = 1
    if use_weights:
        w = graph.edges[edge]["score"]
    i = edge[0]
    j = edge[1]
    if graph.nodes[i]["ndd"]:
        y[i, j, 1] = model.addVar(vtype=GRB.BINARY, obj=w*p, name=f"y[{i},{j},{1}]")
    else:
        for k in range(2, L + 1):
            y[i, j, k] = model.addVar(vtype=GRB.BINARY, obj=w*p**k, name=f"y[{i},{j},{k}]")
        # y[i, j, 1] = model.addVar(vtype=GRB.BINARY, obj=w*p, name=f"y[{i},{j},{1}]", ub=0)
    
def get_cycle_score(graph: nx.DiGraph, cycle: list, use_weights=True) -> float:
    if not use_weights:
        return len(cycle)
    k = len(cycle)
    s = 0
    for i in range(k - 1):
        s += graph[cycle[i]][cycle[i+1]]["score"]
    s += graph[cycle[k - 1]][cycle[0]]["score"]
    return s

def get_cycle_scores(graph: nx.DiGraph, cycles: list[list], use_weights=True) -> dict:
    cycle_scores = {}
    for c, cycle in enumerate(cycles):
        cycle_scores[c] = get_cycle_score(graph=graph, cycle=cycle) if use_weights else len(cycle)
    return cycle_scores

def get_chain_score(graph: nx.DiGraph, chain: list, use_weights=True) -> float:
    if not use_weights:
        return len(chain) - 1
    k = len(chain)
    s = 0
    for i in range(k - 1):
        s += graph[chain[i]][chain[i+1]]["score"]
    return s

def get_chain_scores(graph: nx.DiGraph, chains: list[list], use_weights=True) -> dict:
    chain_scores = {}
    for c, chain in enumerate(chains):
        chain_scores[c] = get_chain_score(graph=graph, chain=chain) if use_weights else len(chain) - 1
    return chain_scores

def add_x_cycle_var(model: gp.Model, x: dict, graph: nx.DiGraph, cycle, c: int, p: float, use_weights=False) -> None:
    
    k = len(cycle)
    w = k

    if use_weights:
        w = get_cycle_score(graph=graph, cycle=cycle)

    x[c] = model.addVar(vtype=GRB.BINARY, obj=w*(p**k), name=f"x[{c}]")

def add_ndd_cap_constr(model: gp.Model, graph: nx.DiGraph, y: dict, node) -> None:
    expr = gp.LinExpr()
    for successor in graph.successors(node):
        expr += y[node, successor, 1]
                
    model.addConstr(expr <= 1, name=f"4c_{node}")

def add_patient_cap_constr_cycle(model: gp.Model, graph: nx.DiGraph, cycles: list, L: int, y: dict, x: dict, node) -> None:
    
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

def add_flow_cons_constr(model: gp.Model, graph: nx.DiGraph, L: int, y: dict, node) -> None:
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

def match_kidneys(graph: nx.DiGraph, K=3, L=4, version="PICEF", p=1, consider_leftover=False, use_weights=False, draw_solution=False) -> float:

    versions = ["PICEF", "HPICEF", "PIEF"]
    assert version in versions, "version invalid"

    if version == "PIEF":
        L = 0

    model = gp.Model("function_solver")
    model.ModelSense = GRB.MAXIMIZE
    # model.params.OutputFlag = 0
    
    y = {}
    x = {}

    if L > 0:
        for edge in graph.edges:
            add_y_vars(model=model, graph=graph, y=y, edge=edge, L=L, p=p, use_weights=use_weights)
    
    if version == "PICEF":

        cycles = list(nx.simple_cycles(graph, length_bound=K))
        print(f"number of cycles of length {K}, or less: {len(cycles)}")

        for c, cycle in enumerate(cycles):
            add_x_cycle_var(model=model, x=x, cycle=cycle, c=c, p=p, graph=graph, use_weights=use_weights)
        
        for node in graph.nodes:

            if graph.nodes[node]["ndd"] and L > 0:
                add_ndd_cap_constr(model=model, graph=graph, y=y, node=node)
            else:
                # print("not ndd")
                add_patient_cap_constr_cycle(model=model, graph=graph, cycles=cycles, L=L, x=x, y=y, node=node)
                add_flow_cons_constr(model=model, graph=graph, L=L, y=y, node=node)

        model.optimize()


        if draw_solution:

            y_vals = {i: y[i].X for i in y}
            x_vals = {i: x[i].X for i in x}

            draw_solution(graph=graph, cycles=cycles, x_vals=x_vals, y_vals=y_vals)

        return model.ObjVal


    else:
        ...
        return 0   
    
def remove_random(graph: nx.DiGraph, frac_ndd: float, frac_edges: float, seed: int) -> nx.DiGraph:

    ndds = [n for n in graph.nodes if graph.nodes[n]["ndd"]]
    print(f"n: {len(graph.nodes)}, ndds: {len(ndds)}, arcs: {len(graph.edges)}")
    
    random.seed(23)

    num_edges_to_remove = int(frac_edges * graph.number_of_edges())
    edges_to_remove = random.sample(list(graph.edges), num_edges_to_remove)
    graph.remove_edges_from(edges_to_remove)

    num_ndds_to_remove = int(frac_ndd * len(ndds))
    ndds_to_remove = random.sample(ndds, num_ndds_to_remove)
    graph.remove_nodes_from(ndds_to_remove)

    ndds = [n for n in graph.nodes if graph.nodes[n]["ndd"]]
    print(f"n: {len(graph.nodes)}, ndds: {len(ndds)}, arcs: {len(graph.edges)}")

    return graph

def macth_kidneys_cycle(graph: nx.DiGraph, cycles: list, cycle_scores: dict=None, chains: list=None, chain_scores: dict=None, use_weights=True) -> dict | float:    

    model = gp.Model()
    model.ModelSense = GRB.MAXIMIZE
    model.params.OutputFlag = 0

    if cycle_scores == None:
        cycle_scores = get_cycle_scores(graph=graph, cycles=cycles, use_weights=use_weights)

    if chain_scores == None:
        chain_scores = get_chain_scores(graph=graph, chains=chains, use_weights=use_weights)
    
    x = {}
    n = len(cycles)


    for c, cycle in enumerate(cycles):
        x[c] = model.addVar(vtype=GRB.BINARY, obj=cycle_scores[c], name=f"x[{c}]")

    for c, chain in enumerate(chains):
        x[n+c] = model.addVar(vtype=GRB.BINARY, obj=chain_scores[c], name=f"x[{n+c}]")

    for node in graph.nodes:
        expr = gp.LinExpr()
        for c, cycle, in enumerate(cycles):
            if node in cycle:
                expr += x[c]
        for c, chain, in enumerate(chains):
            if node in chain:
                expr += x[n+c]
        model.addConstr(expr <= 1, name=f"node {node} cap constr")

    model.optimize()

    x_vals = {i: x[i].X for i in x}

    return x_vals, model.ObjVal

def get_cycles(graph: nx.DiGraph, K: int) -> list:
    cycles = (list(nx.simple_cycles(graph, length_bound=K)))
    valid_cycles = []
    for cycle in cycles:
        cycle_set = set(cycle)
        valid = True
        for node in cycle:
            if not valid:
                break
            if graph.nodes[node]["ndd"]:
                valid = False
                break
            for cycle_other in valid_cycles:
                if cycle_set == set(cycle_other):
                    valid = False
                    break
        if valid:
            valid_cycles.append(cycle)
    return valid_cycles

def get_chains(graph: nx.DiGraph, L: int) -> list:
    chains = []
    ndds = [n for n in graph.nodes if graph.nodes[n]["ndd"]]
    for ndd in ndds:
        all_paths = [p for p in all_simple_paths_from(graph, ndd, L) if len(p)>1]

        for path in all_paths:
            chains.append(path)
    return chains

def all_simple_paths_from(graph: nx.DiGraph, source, L:int):
    def dfs(path):
        yield path
        if len(path)-1 == L:
            return
        for nbr in graph.successors(path[-1]):
            if nbr not in path and not graph.nodes[nbr]["ndd"]:
                yield from dfs(path + [nbr])

    yield from dfs([source])

    