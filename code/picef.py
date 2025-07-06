import networkx as nx
import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from kep_simulator import DynamicPoolSimulator
from static_kidney_utils import*
from data_utils import*

class PICEFSolver:
    def __init__(self, graph: nx.DiGraph, K: int, L: int, use_weights=False,
                 periods: dict=None, multiperiod=True, debug=False, time_limit=None,
                 batch_size=1, prefer_present=True):
        
        self.periods = self.get_relevant_periods(periods=periods, batch_size=batch_size)
        self.all_nodes = self.get_all_nodes(graph=graph, periods=self.periods)
        self.graph = nx.induced_subgraph(G=graph, nbunch=self.all_nodes).copy()
        self.is_ndd = {node: graph.nodes[node]["ndd"] for node in self.all_nodes}
        self.pairs = [node for node in self.all_nodes if not self.is_ndd[node]]
        self.ndds = [node for node in self.all_nodes if self.is_ndd[node]]

        if multiperiod:
            assert periods != None, "need periods for multiperiod model"
            self.graph.add_node(0, ndd=False)
            for node in self.pairs:
                self.graph.add_edge(node, 0, score=0)

        cycles_by_period, period_subgraphs = self.period_subgraphs_and_cycles(graph=self.graph, K=K, multiperiod=multiperiod, periods=self.periods, is_ndd=self.is_ndd)
        self.cycles_by_period = cycles_by_period
        self.period_subgraphs = period_subgraphs
        
        self.K = K
        self.L = L
        self.use_weights = use_weights
        self.multiperiod = multiperiod
        self.prefer_present = prefer_present
        self.debug = debug
        
        if not periods == None:
            self.period_indeces = list(self.periods.keys())
            node_in_period = {}
            for p in self.period_indeces:
                node_in_period[p, 0] = True
                for n in self.all_nodes:
                    node_in_period[p,n] = n in self.periods[p]["pool"]
            self.node_in_period = node_in_period
        
        self.model = gp.Model("picef_solver")
        
        if not debug:
            self.model.params.OutputFlag = 0
        if not time_limit == None:
            self.model.params.TimeLimit = time_limit
        self.model.ModelSense = GRB.MAXIMIZE
            
        self.x = {}
        self.y = {}

        self.model_is_built = False

    @staticmethod
    def period_subgraphs_and_cycles(graph: nx.DiGraph, K: int, periods: dict, is_ndd: dict, multiperiod: bool) -> dict|dict:
        
        cycles_by_period = {}
        period_subgraphs = {}

        if not multiperiod:
            cycles = get_cycles(graph=graph.subgraph([n for n in graph.nodes if not is_ndd[n]]), K=K)
            cycles_by_period[0] = cycles
            period_subgraphs[0] = graph

        else:
            period_indeces = list(periods.keys())
            for p in period_indeces:

                period_pairs = [n for n in periods[p]["pool"] if not is_ndd[n]]
                nodes = list(periods[p]["pool"] + [0])

                cycles = get_cycles(graph=graph.subgraph(period_pairs), K=K)
                period_subgraph = graph.subgraph(nodes)
                
                if p > 0:
                    period_cycles = []
                    for cycle in cycles:
                        is_new = True
                        set_cycle = set(cycle)
                        for t in range(p):
                            for cycle_other in cycles_by_period[t]:
                                if  set_cycle == set(cycle_other):
                                    is_new = False
                                    break
                                if not is_new:
                                    break
                        if is_new:
                            period_cycles.append(cycle)
                    cycles_by_period[p] = period_cycles
                    period_subgraphs[p] = period_subgraph
                else:
                    cycles_by_period[p] = cycles
                    period_subgraphs[p] = period_subgraph

        return cycles_by_period, period_subgraphs

    @staticmethod
    def get_all_nodes(graph: nx.DiGraph, periods: dict=None) -> list:
        if periods == None:
            return graph.nodes
        else:
            all_nodes = []
            for p in periods.keys():
                for node in periods[p]["pool"]:
                    if not node in all_nodes:
                        all_nodes.append(node)
            return all_nodes
        
    @staticmethod
    def get_relevant_periods(periods: dict=None, batch_size: int=1) -> dict:
        if batch_size > 1 and not periods == None:
            period_indeces = list(periods.keys())
            last_period = period_indeces[-1]
            relevant_periods = {}

    
            
            for p in period_indeces:
                if p * batch_size >= last_period:
                    relevant_periods[p] = periods[last_period]
                    return relevant_periods
                else:
                    relevant_periods[p] = periods[p*batch_size]
            
        elif batch_size == 0:
            last_period = list(periods.keys())[-1]
            relevant_periods = {0: periods[0], 1: periods[last_period]}
            return relevant_periods

        else:
            return periods

    def add_y_vars_mp(self) -> None:
        assert not self.model_is_built, "model already built"
        assert self.multiperiod, "model not multiperiod"
        start = time.time()

        y = self.y
        model = self.model
        period_subgraphs = self.period_subgraphs
        period_indeces = self.period_indeces
        is_ndd = self.is_ndd
        use_weights = self.use_weights
        node_in_period = self.node_in_period
        prefer_present = self.prefer_present
        L = self.L

        for p in period_indeces:
            period_subgraph = period_subgraphs[p]
            for edge in period_subgraph.edges:
                
                i = edge[0]
                j = edge[1]

                score = period_subgraph.edges[edge]["score"] if use_weights else 0 if j == 0 else 1
                
                if prefer_present and p == 0:
                    score += 0.001

                if i == 0:
                    pass

                elif is_ndd[i]:
                    if not j == 0:
                        y[i,j,1,p] = model.addVar(vtype=GRB.BINARY, obj=score, name=f"y[{i},{j},{1},{p}]")
                else:
                    if j != 0 and is_ndd[j]:
                        continue
                    
                    elif p > 0 and node_in_period[p-1, i]:
                        y[i,j,1,p] = model.addVar(vtype=GRB.BINARY, obj=score, name=f"y[{i},{j},{1},{p}]")

                    for k in range(2, L+1):
                        y[i,j,k,p] = model.addVar(vtype=GRB.BINARY, obj=score, name=f"y[{i},{j},{k},{p}]")
                    
                    if j == 0:
                        y[i,j,L+1,p] = model.addVar(vtype=GRB.BINARY, obj=score, name=f"y[{i},{j},{L+1},{p}]")

        end = time.time()
        if self.debug:
            print(f"added y multi period vars in {end - start} seconds")

    def add_y_vars_sp(self) -> None:
        assert not self.model_is_built, "model already built"
        assert not self.multiperiod, "model is multi period"
        start = time.time()

        y = self.y
        model = self.model
        graph = self.graph
        is_ndd = self.is_ndd
        use_weights = self.use_weights
        L = self.L

        for edge in graph.edges:
            i = edge[0]
            j = edge[1]
            score = graph.edges[edge]["score"] if use_weights else 1
            if is_ndd[j]:
                continue
            if is_ndd[i]:
                y[i,j,1] = model.addVar(vtype=GRB.BINARY, obj=score, name=f"y[{i},{j},{1}]")
            else:
                for k in range(2, L+1):
                    y[i,j,k] = model.addVar(vtype=GRB.BINARY, obj=score, name=f"y[{i},{j},{k}]")

        end = time.time()
        if self.debug:
            print(f"added y single period vars in {end - start} seconds")

    def add_x_vars_mp(self) -> None:
        assert not self.model_is_built, "model already built"
        assert self.multiperiod, "model not multiperiod"
        start = time.time()

        x = self.x
        model = self.model
        cycles_by_period = self.cycles_by_period
        use_weights = self.use_weights
        graph = self.graph

        for p in self.period_indeces:
            for c, cycle in enumerate(cycles_by_period[p]):
                score = get_cycle_score(graph=graph, cycle=cycle) if use_weights else len(cycle)
                x[c,p] = model.addVar(vtype=GRB.BINARY, obj=score, name=f"x[{c},{p}]")

        end = time.time()
        if self.debug:
            print(f"added x multi period vars in {end - start} seconds")

    def add_x_vars_sp(self) -> None:
        assert not self.model_is_built, "model already built"
        assert not self.multiperiod, "model is multi period"
        start = time.time()

        x = self.x
        model = self.model
        cycles = self.cycles_by_period[0]
        use_weights = self.use_weights
        graph = self.graph

        for c, cycle in enumerate(cycles):
            score = get_cycle_score(graph=graph, cycle=cycle) if use_weights else len(cycle)
            x[c] = model.addVar(vtype=GRB.BINARY, obj=score, name=f"x[{c}]")

        end = time.time()
        if self.debug:
            print(f"added x single period vars in {end - start} seconds")

    def add_capacity_constraints_mp(self) -> None:
        assert self.multiperiod, "model not multiperiod"
        assert not self.model_is_built, "model already built"
        start = time.time()

        periods = self.periods
        is_ndd = self.is_ndd
        y = self.y
        x = self.x
        period_indeces = self.period_indeces
        pairs = self.pairs
        ndds = self.ndds
        L = self.L
        period_subgraphs = self.period_subgraphs
        cycles_by_period = self.cycles_by_period
        model = self.model

        for ndd in ndds:
            cap = gp.LinExpr()
            for p in period_indeces:
                if ndd in periods[p]["pool"]:
                    for successor in period_subgraphs[p].successors(ndd):
                        cap += y[ndd, successor, 1, p]
            model.addConstr(cap <= 1, name=f"capacity ndd {ndd}")
        for node in pairs:
            cap = gp.LinExpr()
            for p in period_indeces:
                if node in periods[p]["pool"]:
                    period_subgraph = period_subgraphs[p]
                    for predecessor in period_subgraph.predecessors(node):
                        if is_ndd[predecessor]:
                            cap += y[predecessor, node, 1, p]
                        else:
                            for k in range(2, L+1):
                                cap += y[predecessor, node, k, p]
                            if p > 0 and predecessor in periods[p-1]["pool"]:
                                cap += y[predecessor, node, 1, p]
                    for c, cycle in enumerate(cycles_by_period[p]):
                        if node in cycle:
                            cap += x[c,p]
            model.addConstr(cap <= 1, name=f"capicity pair {node}")

        end = time.time()
        if self.debug:
            print(f"added multi period capacity constraints in {end - start} seconds")

    def add_capacity_constraints_sp(self) -> None:
        assert not self.multiperiod, "model is multi period"
        assert not self.model_is_built, "model already built"
        start = time.time()

        is_ndd = self.is_ndd
        y = self.y
        x = self.x
        graph = self.graph
        L = self.L
        model = self.model
        cycles_by_period = self.cycles_by_period
        all_nodes = self.all_nodes

        for node in all_nodes:
            if is_ndd[node]:
                cap = gp.LinExpr()
                for successor in graph.successors(node):
                    if not is_ndd[successor]:
                        cap += y[node, successor, 1]
                model.addConstr(cap <= 1, name=f"4c_{node}")
            else:
                cap = gp.LinExpr()
                for c, cycle in enumerate(cycles_by_period[0]):
                    if node in cycle:
                        cap += x[c]
                for predecessor in graph.predecessors(node):
                    if is_ndd[predecessor]:
                        cap += y[predecessor, node, 1]
                    else:
                        for k in range(2, L+1):
                            cap += y[predecessor, node, k]
                model.addConstr(cap <= 1, name=f"4b_{node}")

        end = time.time()
        if self.debug:
            print(f"added single period capacity constraints in {end - start} seconds")
        
    def add_node_capacity_constraint_mp(self, model, node, period_subgraphs, period_indeces, cycles_by_period, y, x, is_ndd, L, node_in_period) -> None:
        assert self.multiperiod, "model not multiperiod"
        assert not self.model_is_built, "model already built"

        expr = gp.LinExpr()

        for p in period_indeces:
            if node_in_period[p, node]:
                period_subgraph = period_subgraphs[p]
                if is_ndd[node]:
                    for successor in period_subgraph.successors(node):
                        if not (successor == 0 or is_ndd[successor]):
                            expr += y[node, successor, 1, p]
                else:
                    for c, cycle in enumerate(cycles_by_period[p]):
                        if node in cycle:
                            expr += x[c, p]
                    for predecessor in period_subgraph.predecessors(node):
                        assert predecessor != 0, "0 node should not be predecessor to anyone"
                        if is_ndd[predecessor]:
                            expr += y[predecessor, node, 1, p]
                        else:
                            if p > 0 and node_in_period[p-1, predecessor]:
                                expr += y[predecessor, node, 1, p]
                            for k in range(2, L+1):
                                expr += y[predecessor, node, k, p]
                    
        model.addConstr(expr <= 1, name=f"mp_cap_{node}")          

    def add_node_flow_conservation_constraint_mp(self, model, node, period_subgraphs, y, is_ndd, L, node_in_period, period_indeces) -> None:
        assert self.multiperiod, "model not multiperiod"
        assert not self.model_is_built, "model already built"

        if is_ndd[node]:
            return
        
        for p in period_indeces:
            if node_in_period[p, node]:
                period_subgraph = period_subgraphs[p]
                for k in range(1, L+1):
                    expr = gp.LinExpr()
                    for predecessor in period_subgraph.predecessors(node):
                        assert predecessor != 0, "0 node should not be predecessor to anyone"
                        if k == 1:
                            if is_ndd[predecessor] or p > 0 and node_in_period[p-1, predecessor]:
                                expr += y[predecessor, node, k, p]
                        elif not is_ndd[predecessor]:
                            expr += y[predecessor, node, k, p]
                    for successor in period_subgraph.successors(node):
                        if successor != 0 and is_ndd[successor]:
                            continue
                        if k < L:
                            expr -= y[node, successor, k+1, p]
                    if k == L:
                        expr -= y[node, 0, k+1, p]
                    model.addConstr(expr == 0, name=f"flow_n_{node}_p_{p}_k_{k}")

    def add_node_bridge_constraint(self, model, node, period_subgraphs, y, is_ndd, L, node_in_period, period_indeces) -> None:
        assert self.multiperiod, "model not multiperiod"
        assert not self.model_is_built, "model already built"

        if is_ndd[node]:
            return
        
        for p in period_indeces:
            if p == 0:
                continue
            if node_in_period[p, node] and node_in_period[p-1, node]:
                period_subgraph = period_subgraphs[p]
                expr = gp.LinExpr()
                for successor in period_subgraph.successors(node):
                    if successor != 0 and is_ndd[successor]:
                            continue
                    expr += y[node, successor, 1, p]
                for k in range(1, L+2):
                    if k == 1:
                        if p > 1 and node_in_period[p-2, node]:
                            expr -= y[node, 0, k, p-1]
                    else:
                        expr -= y[node, 0, k, p-1]
                model.addConstr(expr == 0, name=f"bridge_n_{node}_p_{p}")

    def add_flow_conservation_constraints_mp(self) -> None:
        assert not self.model_is_built, "model already built"
        assert self.multiperiod, "model not multiperiod"

        start = time.time()
        periods = self.periods
        is_ndd = self.is_ndd
        y = self.y
        period_indeces = self.period_indeces
        pairs = self.pairs
        L = self.L
        period_subgraphs = self.period_subgraphs
        model = self.model

        for node in pairs:
            for p in period_indeces:
                if node in periods[p]["pool"]:
                    period_subgraph = period_subgraphs[p]
                    k = 1
                    lhs = gp.LinExpr()
                    for predecessor in period_subgraph.predecessors(node):
                        if is_ndd[predecessor]:
                            lhs += y[predecessor, node, k, p]
                        elif p > 0 and predecessor in periods[p-1]["pool"]:
                            lhs += y[predecessor, node, k, p]
                    for successor in period_subgraph.successors(node):
                        lhs -= y[node, successor, k+1, p]
                    model.addConstr(lhs == 0, name=f"flow cons node: {node}, period: {p}, position: {k}")
                    
                    for k in range(2, L):
                        lhs = gp.LinExpr()
                        for predecessor in period_subgraph.predecessors(node):
                            if not is_ndd[predecessor]:
                                lhs += y[predecessor, node, k, p]
                        for successor in period_subgraph.successors(node):
                            lhs -= y[node, successor, k+1, p]
                        model.addConstr(lhs == 0, name=f"flow cons node: {node}, period: {p}, position: {k}")

                    k = L
                    lhs = gp.LinExpr()
                    for predecessor in period_subgraph.predecessors(node):
                        if not is_ndd[predecessor]:
                            lhs += y[predecessor, node, k, p]
                    lhs -= y[node, 0, k+1, p]
                    model.addConstr(lhs >= 0, name=f"flow cons node: {node}, period: {p}, position: {k}")

        end = time.time()
        if self.debug:
            print(f"added multi period flow conservation constraints in {end - start} seconds")

    def add_flow_conservation_constraints_sp(self) -> None:
        assert not self.multiperiod, "model is multi period"
        assert not self.model_is_built, "model already built"

        start = time.time()
        y = self.y
        is_ndd = self.is_ndd
        graph = self.graph
        model = self.model
        pairs = self.pairs
        L = self.L

        for node in pairs:

            for k in range(1, L):
                expr = gp.LinExpr()
                for predecessor in graph.predecessors(node):
                    if is_ndd[predecessor]:
                        if k == 1:
                            expr += y[predecessor, node, k]
                    else:
                        if k > 1:
                            expr += y[predecessor, node, k]
                for successor in graph.successors(node):
                    if not is_ndd[successor]:
                        expr -= y[node, successor, k+1]
                model.addConstr(expr >= 0, name=f"4d_{node}_{k}")
        
        end = time.time()
        if self.debug:
            print(f"added single period flow conservation constraints in {end - start} seconds")

        model.optimize()

    def add_bridge_constraints(self) -> None:
        assert not self.model_is_built, "model already built"
        assert self.multiperiod, "model not multiperiod"

        start = time.time()
        periods = self.periods
        is_ndd = self.is_ndd
        y = self.y
        period_indeces = self.period_indeces
        pairs = self.pairs
        L = self.L
        period_subgraphs = self.period_subgraphs
        model = self.model

        for node in pairs:
            for p in period_indeces:
                
                if p == 0:
                    continue
                if node in periods[p]["pool"] and node in periods[p-1]["pool"]:
                    period_subgraph = period_subgraphs[p]
                    lhs = gp.LinExpr()
                    for successor in period_subgraph.successors(node):
                        if successor == 0 or not is_ndd[successor]:
                            lhs += y[node, successor, 1, p]
                    for k in range(2, L+2):
                        lhs -= y[node, 0, k, p-1]
                    if p > 1 and node in periods[p-2]["pool"]:
                        lhs -= y[node, 0, 1, p-1]
                    model.addConstr(lhs == 0, name=f"bridge node: {node}, time: {p}")

        end = time.time()
        if self.debug:
            print(f"added multi period bridge constraints in {end - start} seconds")

    def add_node_capacity_constraint_sp(self, model, node, graph, y, x, cycles, L, is_ndd) -> None:
        assert not self.model_is_built, "model already built"
        assert not self.multiperiod, "model is multi period"

        if is_ndd[node]:
            expr = gp.LinExpr()
            for successor in graph.successors(node):
                if not is_ndd[successor]:
                    expr += y[node, successor, 1]
            model.addConstr(expr <= 1, name=f"4c_{node}")

        else:
            expr =  gp.LinExpr()
            for c, cycle in enumerate(cycles):
                if node in cycle:
                    expr += x[c]
            for predecessor in graph.predecessors(node):
                if is_ndd[predecessor]:
                    expr += y[predecessor, node, 1]
                else:
                    for k in range(2, L + 1):
                        expr += y[predecessor, node, k]
                
            model.addConstr(expr <= 1, name=f"4b_{node}")

    def add_node_flow_conservation_constraint_sp(self, model, node, graph, y, L, is_ndd) -> None:

        assert not self.model_is_built, "model already built"
        assert not self.multiperiod, "model is multi period"

        if is_ndd[node]:
            return
        else:
        
            for k in range(1, L):
                expr = gp.LinExpr()
                for predecessor in graph.predecessors(node):
                    if is_ndd[predecessor]:
                        if k == 1:
                            expr += y[predecessor, node, 1]
                    else:
                        if k > 1:
                            expr += y[predecessor, node, k]
                for successor in graph.successors(node):
                    if not is_ndd[successor]:
                        expr -= y[node, successor, k+1]
                model.addConstr(expr >= 0, name=f"4d_{node}_{k}")

    def build_model(self):
        if self.multiperiod:
            self.add_x_vars_mp()
            self.add_y_vars_mp()
            for node in self.all_nodes:
                self.add_node_capacity_constraint_mp(model=self.model, y=self.y, x=self.x, is_ndd=self.is_ndd, 
                                                     node=node, node_in_period=self.node_in_period, L=self.L, 
                                                     cycles_by_period=self.cycles_by_period, period_indeces=self.period_indeces, 
                                                     period_subgraphs=self.period_subgraphs)

            for node in self.pairs:
                self.add_node_flow_conservation_constraint_mp(model=self.model, y=self.y, node=node, L=self.L, 
                                                              node_in_period=self.node_in_period, period_indeces=self.period_indeces, 
                                                              period_subgraphs=self.period_subgraphs, is_ndd=self.is_ndd)
                
                self.add_node_bridge_constraint(model=self.model, y=self.y, L=self.L, period_indeces=self.period_indeces, 
                                                period_subgraphs=self.period_subgraphs, is_ndd=self.is_ndd, 
                                                node_in_period=self.node_in_period, node=node, )
                
            self.model_is_built = True
        else:
            self.add_x_vars_sp()
            self.add_y_vars_sp()
            for node in self.all_nodes:
                self.add_node_capacity_constraint_sp(model=self.model, node=node, graph=self.graph, y=self.y, x=self.x, cycles=self.cycles_by_period[0], L=self.L, is_ndd=self.is_ndd)
                self.add_node_flow_conservation_constraint_sp(model=self.model, node=node, graph=self.graph, y=self.y, L=self.L, is_ndd=self.is_ndd)
                
            self.model_is_built = True
    
    def optimize(self, masked_chain: list=None, masked_cycle: list=None, time_limit=None, pooling=False, force_end_chain=False) -> float:
        if not self.model_is_built:
            self.build_model()

        if not time_limit == None:
            self.model.params.TimeLimit = time_limit

        x = self.x
        y = self.y
        restricted_vars = []

        if not masked_chain == None:
            assert masked_cycle == None, "only one cycle or chain can be masked"

            for k in range(1, len(masked_chain)):
                if self.multiperiod:
                    restricted_vars.append(y[masked_chain[k-1], masked_chain[k], k, 0])
                else:
                    restricted_vars.append(y[masked_chain[k-1], masked_chain[k], k])
            
            if self.multiperiod and force_end_chain:
                restricted_vars.append(y[masked_chain[-1], 0, len(masked_chain), 0])
        
        if not masked_cycle == None:
            assert masked_chain == None, "only one cycle or chain can be masked"
            masked_set = set(masked_cycle)
            for c, cycle in enumerate(self.cycles_by_period[0]):
                if set(cycle) == masked_set:
                    if self.multiperiod:
                        restricted_vars.append(x[c, 0])
                    else:
                        restricted_vars.append(x[c])
                    break

        for restricted_var in restricted_vars:
            restricted_var.LB = 1
        
        self.model.optimize()

        obj = self.model.ObjVal

        for restricted_var in restricted_vars:
            restricted_var.LB = 0

        return obj
    
    def selected_cycles(self) -> list:
        x_vals = {i: self.x[i].X for i in self.x}
        if self.multiperiod:
            selected_cycles = [set(cycle) for c, cycle in enumerate(self.cycles_by_period[0]) if x_vals[c, 0] > 0.5]
            return selected_cycles
        else:
            selected_cycles = [set(cycle) for c, cycle in enumerate(self.cycles_by_period[0]) if x_vals[c] > 0.5]
            return selected_cycles

    def selected_chains(self) -> list:
        if self.multiperiod:
            return self.selected_chains_mp()
        else:
            return self.selected_chains_sp()
    
    def selected_chains_mp(self) -> list:
        assert self.multiperiod, "model not multiperiod"

        selected_chains = []
        graph = self.period_subgraphs[0]
        y_vals = {i: self.y[i].X for i in self.y}

        for ndd in self.ndds:
            if not self.node_in_period[0, ndd]:
                continue
            chain = [ndd]
            used = False
            for successor in graph.successors(ndd):
                if successor == 0:
                    continue
                if self.is_ndd[successor]:
                    continue
                if y_vals[ndd, successor, 1, 0] > 0.5:
                    chain.append(successor)
                    used = True
            if used:
                for k in range(2, self.L+1):
                    ended = True
                    for successor in graph.successors(chain[-1]):
                        if successor == 0:
                            if y_vals[chain[-1], successor, k, 0] > 0.5:
                                break
                        elif not self.is_ndd[successor]:
                            if y_vals[chain[-1], successor, k, 0] > 0.5:
                                chain.append(successor)
                                ended = False
                                break
                    if ended:
                        selected_chains.append(chain)
                        break
        return selected_chains
    
    def selected_chains_sp(self) -> list:
        assert not self.multiperiod, "model is multiperiod"

        y_vals = {i: self.y[i].X for i in self.y}
        selected_chains = []

        graph = self.graph
        for ndd in self.ndds:
            chain = [ndd]
            used = False
            for successor in graph.successors(ndd):
                if not self.is_ndd[successor]:
                    if y_vals[ndd, successor, 1] > 0.5:
                        used = True
                        chain.append(successor)
            if used:
                for k in range(2, self.L+1):
                    ended = True
                    for successor in graph.successors(chain[-1]):
                        if not self.is_ndd[successor]:
                            if y_vals[chain[-1], successor, k] > 0.5:
                                ended = False
                                chain.append(successor)
                                break
                    if ended:
                        selected_chains.append(chain)
                        break
                                
        return selected_chains

                
def main():
    ...
    n = 1
    lookahead = 12
    K = 3
    L = 2
    use_weights = False

    starting_pool_size = 30
    arrival_rate = 15
    departure_rate = 0.0
    renege_rate = 0
    
    path = f"/Users/barnitorzsok/Documents/GitHub/Thesis/data/simulation/converted/simulation_{n}.pkl"
    kep_simulator = DynamicPoolSimulator(path=path, seed=6, init_pool_size=starting_pool_size, arrival_rate=arrival_rate, departure_rate=departure_rate, renege_rate=renege_rate)
    periods = kep_simulator.sample_scenarios(n_samples=1, lookahead=lookahead)[0]

    solver = PICEFSolver(graph=kep_simulator.full_graph, K=K, L=L,use_weights=use_weights, multiperiod=True, debug=True, periods=periods, batch_size=1)
    solver.optimize()





if __name__ == "__main__":
    main()
    