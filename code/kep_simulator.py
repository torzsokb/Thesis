import networkx as nx
import numpy as np
import random
from data_utils import load_graph_from_pickle

class DynamicPoolSimulator:

    def __init__(self, path: str, seed: int, 
                 init_pool_size: int, arrival_rate: float, departure_rate: float, renege_rate: float):
        
        self.full_graph = load_graph_from_pickle(path=path)
        self.config_data = self.full_graph.graph
        self.init_pool_size = init_pool_size
        self.arrival_rate = arrival_rate
        self.departure_rate = departure_rate
        self.renegere_rate = renege_rate
        self.states = ["active", "inactive"]

        for node in self.full_graph.nodes:
            self.full_graph.nodes[node]["state"] = "inactive"

        np.random.seed(seed)
        random.seed(seed)


    def get_nodes(self, state: str) -> list:
        assert state in self.states, "invalid state"
        return [n for n in self.full_graph.nodes if self.full_graph.nodes[n]["state"] == state]
    
    def dynamic_exchange_pool(self, experiment_duration) -> dict:
        starting_pool = self.get_starting_pool()
        periods = {0: {"pool": starting_pool}}
        
        departures = self.get_departures(pool=starting_pool)
        
        for p in range(1, experiment_duration+1):
            arrivals = self.get_arrivals(unused_nodes=self.get_nodes("inactive"))
            new_pool = self.get_new_pool(starting_pool=periods[p-1]["pool"], arrivals=arrivals, departures=departures, real_pool=True)
            period = {"pool": new_pool, "arrivals": arrivals, "departures": departures}
            periods[p] = period
            departures = self.get_departures(pool=new_pool)

        return periods
    
    def sample_scenarios(self, lookahead, n_samples, starting_pool) -> list:
        scenarios = []

        for n in range(n_samples):
            scenarios.append(self.sample_scenario(lookahead=lookahead, starting_pool=starting_pool))

        return scenarios
    
    def sample_scenario(self, lookahead, starting_pool) -> dict:
        periods = {0: {"pool": starting_pool}}
        departures = self.get_departures(starting_pool)
        unused_nodes = self.get_nodes("inactive")

        for p in range(1, lookahead+1):
            arrivals = self.get_arrivals(unused_nodes=unused_nodes)
            new_pool = self.get_new_pool(starting_pool=periods[p-1]["pool"], arrivals=arrivals, departures=departures, real_pool=False, unused_nodes=unused_nodes)
            period = {"pool": new_pool, "arrivals": arrivals, "departures": departures}
            periods[p] = period
            departures = self.get_departures(pool=new_pool)

        return periods

    def get_starting_pool(self) -> list:
        starting_pool = random.sample(self.get_nodes("inactive"), self.init_pool_size)
        for node in starting_pool:
            self.full_graph.nodes[node]["state"] = "active"
        return starting_pool

    def get_departures(self, pool: list) -> list:
        departures = []

        if pool == None or len(pool) == 0:
            return departures

        for node in pool:
            r = random.random()
            if r < self.departure_rate:
                departures.append(node)

        return departures
    
    def get_arrivals(self, unused_nodes: list) -> list:
        num_arrivals = np.random.poisson(lam=self.arrival_rate)
        arrivals = random.sample(unused_nodes, num_arrivals)
        return arrivals
    
    def get_new_pool(self, starting_pool: list, arrivals: list, departures: list, real_pool: bool, unused_nodes: list=None) -> list:
        new_pool = []
        for node in starting_pool:
            if node not in departures:
                new_pool.append(node)
        for node in arrivals:
            new_pool.append(node)
            if real_pool:
                self.full_graph.nodes[node]["state"] = "active"
            else:
                assert unused_nodes != None
                assert node in unused_nodes
                unused_nodes.remove(node)

        return new_pool

def main():
    ...


if __name__ == "__main__":
    main()