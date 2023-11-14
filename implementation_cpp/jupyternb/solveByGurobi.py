import gurobipy as gp
import os

# curDir = os.path.dirname(os.path.realpath(__file__))

presolvedDir = "/home/twh/work/PDLP/implementation_cpp/data/cache/presolved"
logDir = "/home/twh/work/PDLP/implementation_cpp/log/gurobi"
# find all files in presolvedDir end with .mps and not start with pgrk_
mpsFiles = [
    f
    for f in os.listdir(presolvedDir)
    if f.endswith(".mps") and not f.startswith("pgrk_")
]

for mps in mpsFiles:
    m = gp.read(os.path.join(presolvedDir, mps))
    m.optimize()
    m.write(os.path.join(logDir, mps.replace(".mps", ".sol")))
    print(
        "Successfully solve "
        + mps
        + " by gurobi, write to "
        + mps.replace(".mps", ".sol")
        + "."
    )
