import gurobipy as gp
import os

# curDir = os.path.dirname(os.path.realpath(__file__))

presolvedDir = "/home/twh/data_manage/cache/presolved"
logDir = "/home/twh/work/PDLP/implementation_cpp/log/gurobi"
# find all files in presolvedDir end with .mps and not start with pgrk_
mpsFiles = [
    f for f in os.listdir(presolvedDir) if f.endswith(".mps") and f.startswith("pgrk_")
]

# # find all files in presolvedDir end with .mps
# lis = [
#     "rail507",
#     "neos-933638",
#     "satellites3-40-fs",
#     "ns1952667",
#     "neos-4300652-rahue",
#     "app1-2",
#     "triptim1",
#     "uccase9",
#     "ns1696083",
#     "lectsched-3",
# ]
# mpsFiles = [f + ".mps" for f in lis]

for mps in mpsFiles:
    m = gp.read(os.path.join(presolvedDir, mps))
    # set barrier algorithm
    m.Params.method = 2
    # set time limit
    m.Params.TimeLimit = 3600 * 3
    # set log file
    m.Params.LogFile = os.path.join(logDir, mps.replace(".mps", ".log"))
    # disable crossover
    m.Params.Crossover = 0
    m.optimize()
    # m.write(os.path.join(logDir, mps.replace(".mps", ".sol")))
    print(
        "Successfully solve "
        + mps
        + " by gurobi, write to "
        + mps.replace(".mps", ".sol")
        + "."
    )
