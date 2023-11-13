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
    # get %M and %S
    current_time = datetime.now().strftime("%M-%S")

    # write the model to a mps file, set filename as pagerank_$n_.mps
    # generate a random number in [0, 1]
    rand = int(100 * np.random.rand())
    filename = "graph_{:.1e}_{}_{}.txt".format(n, current_time, rand)
    print("write in", filename)
    n = int(n)
    G = nx.barabasi_albert_graph(n, m)

    # get the adjacency matrix
    A = nx.adjacency_matrix(G)

    # scale the matrix so that each column sum to 1
    A /= A.sum(axis=0)

    rows, cols = A.nonzero()

    # write A to txt file,in the format of, row, col, value
    with open(pagerank_path + filename, "w") as f:
        for row, col in zip(rows, cols):
            f.write("%d %d %.16f\n" % (row, col, A[row, col]))

    return


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1])
    m = int(sys.argv[2])

    for i in range(5):
        generate_graph(n, m, i)
