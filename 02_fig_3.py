import networkx as nx
from implementation import affective_polarization as ap
from implementation import synthetic_experiments as data

# Specifications for the experiment.
nnodes = 50                                 # Network size
nedges = 610                                # Number of edges
niter = 100                                 # Number of repetitions

# In this experiment, we change the disagreement--hostility correlation by updating the opinion 
# (and thus the disagreement vector), while the hostility values remain the same.

# The tuples below determine the opinion distribution. There are two values per tuple:
#   1) Number opinion values to replace: Determines the number of opinion values to be replaced.
#      Passed to 'data.make_opinion()' as the parameter 'val'.
#   2) Opinion replacement values: We replace the first i values in the opinion vector o by a
#      specified value. Passed to 'data.make_opinion()' as the parameter 'i'. 
steps = ((3, 0.01), (3, 0.125), (3, 0.25), (3, 0.5), (3, 1.0))                   

results = []
for count in range(niter):

    print(count)

    # Create the network.
    G = nx.gnm_random_graph(n=nnodes, m=nedges)
    Q = ap._ge_Q(network=G)
    o = data.make_opinion(size=nnodes, i=steps[-1][0], val=steps[-1][1])
    dis = data.disagreement(o, G)
    hos = data.hostility(disagree=dis, intv=0.25, positive=True)

    # Test different opinion values (= different disagreement distributions).
    for index, (i, val) in enumerate(steps):
        o = data.make_opinion(size=nnodes, i=i, val=val)
        d = data.disagreement(o, G)
        results.append(data.calculate_measures(index=index, network=G, Q=Q, opinions=o, disagree=d, hostile=hos))
        
# Summarize and save results.
columns = ["network", "pearson", "alpha", "avg_likeminded", "avg_crosscutting", "emd_blue", "emd_red", "sai", "pole"]
cis = data.calculate_confidence_intervals(results=results, columns=columns, by="network", niter=niter)
cis.to_csv("fig_3.csv", index=False)
