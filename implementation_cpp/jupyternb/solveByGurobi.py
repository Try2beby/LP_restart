import gurobipy as gp
import os

# curDir = os.path.dirname(os.path.realpath(__file__))

dataDir = "/home/twh/work/LP_tests/data/gendata"
logDir = "/home/twh/work/PDLP/implementation_cpp/log/gurobi"
# find all files in presolvedDir end with .mps and not start with pgrk_
mpsFiles = [
    f for f in os.listdir(dataDir) if f.endswith(".mps") and f.startswith("trans")
]

# # find all files in presolvedDir end with .mps and start with pgrk_
# mpsFiles = [
#     f
#     for f in os.listdir(presolvedDir)
#     if f.endswith(".mps") and f.startswith("pgrk_")
# ]

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

print("Solve " + str(len(mpsFiles)) + " mps files by gurobi.")
print(mpsFiles)

for mps in mpsFiles:
    # if mps already in logDir, continue
    if mps.replace(".mps", ".log") in os.listdir(logDir):
        print(f"Problem {mps} has been solved.")
        continue

    m = gp.read(os.path.join(dataDir, mps))
    # set barrier algorithm
    m.Params.method = 2
    # set time limit
    m.Params.TimeLimit = 3600
    # set log file
    m.Params.LogFile = os.path.join(logDir, mps.replace(".mps", ".log"))
    # disable crossover
    m.Params.Crossover = 1
    m.Params.Threads = 1

    m.optimize()
    # m.write(os.path.join(logDir, mps.replace(".mps", ".sol")))
    print(
        "Successfully solve "
        + mps
        + " by gurobi, write to "
        + mps.replace(".mps", ".log")
        + "."
    )
