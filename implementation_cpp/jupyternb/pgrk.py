import networkx as nx
import numpy as np
import gurobipy as gp
import scipy.sparse as sp
import os
from datetime import datetime

np.random.seed(0)

pagerank_path = "../data/pagerank/"
if not os.path.exists(pagerank_path):
    os.makedirs(pagerank_path)


def generate_graph(n, m, idx):
    # create a Barabási–Albert graph
    # n is the number of nodes, m is the number of edges to attach from a new node to existing nodes
    print("generating graph with n = %.1e, m = %d\n" % (n, m))
    n = int(n)
    G = nx.barabasi_albert_graph(n, m)

    # get the adjacency matrix
    A = nx.adjacency_matrix(G)

    # scale the matrix so that each column sum to 1
    A /= A.sum(axis=0)

    process(A, 0.85)

    return


def process(Net, _lambda=0.85):
    n = Net.shape[0]
    print("Net shape:", Net.shape)

    # define a model
    m = gp.Model("B-A")
    # add variables
    x = m.addMVar(shape=n, lb=0, name="x")

    # add constraints
    m.addConstr(x.sum() == 1)
    m.addConstr(_lambda * Net @ x + (1 - _lambda) / n <= x)

    # set objective
    m.setObjective(0, gp.GRB.MINIMIZE)

    # get %M and %S
    current_time = datetime.now().strftime("%M-%S")

    # write the model to a mps file, set filename as pagerank_$n_.mps
    # generate a random number in [0, 1]
    rand = int(100 * np.random.rand())

    m.write("pgrk_{}_{}_{}.mps".format(n, current_time, rand))

    # print done
    print("Done!")


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1])
    m = int(sys.argv[2])

    for i in range(5):
        generate_graph(n, m, i)
