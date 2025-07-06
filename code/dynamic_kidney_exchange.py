import networkx as nx
from static_kidney_utils import*
import copy
from kep_simulator import DynamicPoolSimulator
import json
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np
from picef import PICEFSolver
import time
    
class DynamicKEPSolver:

    def __init__(self, kep_simulator: DynamicPoolSimulator, K: int, L: int, lookahead: int, n_samples: int, experiment_duration: int, batch_size: int=1, enable_bridge_donors=False, use_weights=False, multiperiod=False, delta: float=0):
        
        self.kep_simulator = kep_simulator
        self.K = K
        self.L = L
        self.lookahead = lookahead
        self.n_samples = n_samples
        self.experiment_duration = experiment_duration
        self.batch_size = batch_size
        self.enable_bridge_donors = enable_bridge_donors
        self.use_weights = use_weights
        self.multiperiod = multiperiod
        self.delta = delta
        self.experiment_data = kep_simulator.dynamic_exchange_pool(experiment_duration=experiment_duration)
        self.experiment_graph = kep_simulator.full_graph.copy()
        self.current_pool = self.experiment_data[0]["pool"]
        self.matched_nodes = []
        self.bridge_donors =  []
        self.cumulative_obj = 0

    
    def get_scores(self, cycles: list, chains: list, scenarios: list) -> dict|dict:
        raise NotImplementedError("Subclasses must implement this method")
    
    def match_patients(self, period: int):
        start = time.time()
        pool = self.current_pool
        pool_size = len(pool)
        n_ndds = len([n for n in pool if self.experiment_graph.nodes[n]["ndd"]])
        scenarios = self.kep_simulator.sample_scenarios(lookahead=self.lookahead, n_samples=self.n_samples, starting_pool=pool)
        graph = self.experiment_graph.subgraph(pool)

        cycles = get_cycles(graph=graph, K=self.K)
        chains = get_chains(graph=graph, L=self.L)

        cycle_scores, chain_scores = self.get_scores(chains=chains, cycles=cycles, scenarios=scenarios)
        solution, val = macth_kidneys_cycle(graph=graph, cycles=cycles, chains=chains, chain_scores=chain_scores, cycle_scores=cycle_scores)

        selected_cycles = [cycle for c, cycle in enumerate(cycles) if solution[c] > 0.99]
        selected_chains = [chain for c, chain in enumerate(chains) if solution[len(cycles) + c] > 0.99]
        end = time.time()

        myopic_obj = self.remove_matched_nodes(selected_cycles=selected_cycles, selected_chains=selected_chains, period=period)
        self.cumulative_obj += myopic_obj
        print(f"month: {period}\tpool size: {pool_size} ({n_ndds})\t cycles: {len(cycles)} ({len(selected_cycles)})\tchains: {len(chains)} ({len(selected_chains)})\tobj: {myopic_obj:.2f} ({self.cumulative_obj:.2f})\ttime: {(end - start):.2f}")
        return myopic_obj
        

    def remove_matched_nodes(self, selected_cycles: list, selected_chains: list, period: int) -> float:
        
        matched_nodes = []
        bridge_donors = []
        
        myopic_obj = 0

        for cycle in selected_cycles:
            myopic_obj += get_cycle_score(graph=self.experiment_graph, cycle=cycle, use_weights=self.use_weights)
            for node in cycle:
                matched_nodes.append(node)

        for chain in selected_chains:
            myopic_obj += get_chain_score(graph=self.experiment_graph, chain=chain, use_weights=self.use_weights)
            for node in chain:
                matched_nodes.append(node)
            if self.enable_bridge_donors:
                matched_nodes.remove(chain[-1])
                bridge_donors.append(chain[-1])

        for node in matched_nodes:
            self.matched_nodes.append(node)
        
        if period < self.experiment_duration:
            new_pool = []
            for node in self.experiment_data[period+1]["pool"]:
                if not node in self.matched_nodes:
                    new_pool.append(node)
            self.current_pool = new_pool
            
            
        if self.enable_bridge_donors:
            for node in bridge_donors:
                self.experiment_graph.nodes[node]["ndd"] = True

        return myopic_obj

    def run_experiment(self):
        # for p in range(1, self.experiment_duration+1):
        #     print(len(self.experiment_data[p]["pool"]))
        # oracle_sol = 0

        print(f"method: {type(self)}\tmultiperiod: {self.multiperiod}\tlookahead: {self.lookahead} ({self.batch_size})\tdelta: {self.delta}\tK: {self.K} L: {self.L}")
        
        solver = PICEFSolver(graph=self.experiment_graph, K=self.K, L=self.L, use_weights=self.use_weights, periods=self.experiment_data, multiperiod=False, prefer_present=False)
        oracle_sol = solver.optimize()
        vals = []
        for p in range(self.experiment_duration + 1):
            val = self.match_patients(period=p)
            vals.append(val)
        print(f"average: {np.mean(vals):.2f}\ttotal: {sum(vals):.2f}\toracle: {oracle_sol:.2f}")
            

    

class APST1(DynamicKEPSolver):

    def __init__(self, kep_simulator, K, L, lookahead, n_samples, experiment_duration, batch_size = 1, enable_bridge_donors=False, use_weights=False, multiperiod=False, delta=0):
        super().__init__(kep_simulator, K, L, lookahead, n_samples, experiment_duration, batch_size, enable_bridge_donors, use_weights, multiperiod, delta)

    def get_scores(self, cycles, chains, scenarios) -> dict|dict:

        cycle_scores = {c: 0 for c, cycle in enumerate(cycles)}
        chain_scores = {c: 0 for c, chain in enumerate(chains)}

        for scenario in scenarios:

            picef_solver = PICEFSolver(graph=self.experiment_graph, K=self.K, L=self.L, use_weights=self.use_weights, multiperiod=self.multiperiod, debug=False, periods=scenario, batch_size=self.batch_size)
            val = picef_solver.optimize()
            
            selected_cycles = picef_solver.selected_cycles()
            selected_chains = picef_solver.selected_chains()

            for c, cycle in enumerate(cycles):
                if set(cycle) in selected_cycles:
                    cycle_scores[c] += val
                else:
                    cycle_scores[c] -= self.delta
            for c, chain in enumerate(chains):
                if chain in selected_chains:
                    chain_scores[c] += val
                else:
                    chain_scores[c] -= self.delta

            picef_solver.model.dispose()

        return cycle_scores, chain_scores

    
    
class APST2(DynamicKEPSolver):

    def __init__(self, kep_simulator, K, L, lookahead, n_samples, experiment_duration, batch_size = 1, enable_bridge_donors=False, use_weights=False, multiperiod=False, delta=0, force_end_chain=True, noise_scale=0):
        super().__init__(kep_simulator, K, L, lookahead, n_samples, experiment_duration, batch_size, enable_bridge_donors, use_weights, multiperiod, delta)
        self.force_end_chain = force_end_chain
        self.noise_scale = noise_scale

    
    def get_scores(self, cycles, chains, scenarios) -> dict|dict:

        cycle_scores = {c: 0 for c, cycle in enumerate(cycles)}
        chain_scores = {c: 0 for c, chain in enumerate(chains)}

        for scenario in scenarios:
            picef_solver = PICEFSolver(graph=self.experiment_graph, K=self.K, L=self.L, use_weights=self.use_weights, multiperiod=self.multiperiod, debug=False, periods=scenario, batch_size=self.batch_size)
            val = picef_solver.optimize()
            for c, cycle in enumerate(cycles):

                c_sol = picef_solver.optimize(masked_cycle=cycle)
                if self.noise_scale > 0:
                    c_sol += np.random.normal(loc=0, scale=val*self.noise_scale)

                if self.delta > 0:
                    s = get_cycle_score(graph=picef_solver.graph, cycle=cycle, use_weights=self.use_weights)
                    # s = c_sol
                    s -= self.delta * (val - c_sol) ** 2
                    cycle_scores[c] += s
                else:
                    cycle_scores[c] += c_sol
                
            for c, chain in enumerate(chains):
                c_sol = picef_solver.optimize(masked_chain=chain, force_end_chain=self.force_end_chain)
                if self.noise_scale > 0:
                    c_sol += np.random.normal(loc=0, scale=val*self.noise_scale)
                if self.delta > 0:
                    s = get_chain_score(graph=picef_solver.graph, chain=chain, use_weights=self.use_weights)
                    # s = c_sol
                    s -= self.delta * (val - c_sol) ** 2
                    chain_scores[c] += s
                else:
                    chain_scores[c] += c_sol
                
       
        return cycle_scores, chain_scores


def main():
    n = 0
    init_pool_size = 20
    arrival_rate = 10
    departure_rate = 0.04
    renege_rate = 0 #not used: it is modeled as simple departure
    seed = 9
    K = 3
    L = 3
    lookahead = 6
    n_samples = 10
    delta = 0
    use_weights = False
    enable_bridge_donors = True
    multiperiod = False
    experiment_duration = 36
    batch_size = 3
    path = f"data/generated/converted_{n}.pkl"

    kep_simulator = DynamicPoolSimulator(path=path, seed=seed, init_pool_size=init_pool_size, arrival_rate=arrival_rate, departure_rate=departure_rate, renege_rate=renege_rate)
    kep_simulator.full_graph.remove_node(0)
    # solver = APST1(kep_simulator=kep_simulator, K=K, L=L, lookahead=lookahead, n_samples=n_samples, experiment_duration=experiment_duration, enable_bridge_donors=enable_bridge_donors, delta=delta, use_weights=use_weights, batch_size=batch_size, multiperiod=multiperiod)
    solver = APST2(kep_simulator=kep_simulator, K=K, L=L, lookahead=lookahead, n_samples=n_samples, experiment_duration=experiment_duration, enable_bridge_donors=enable_bridge_donors, use_weights=use_weights, multiperiod=multiperiod, batch_size=batch_size, force_end_chain=True, delta=delta)

   
    solver.run_experiment()

if __name__ == "__main__":
    main()
    