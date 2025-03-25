import networkx as nx
from implementation import affective_polarization as ap  
from implementation import synthetic_experiments as data

# Specifications for the experiment.
nnodes = 50                                 # Network size
nedges = 610                                # Number of edges
niter = 100                                 # Number of repetitions

# In this experiment, we change the disagreement--hostility relation by changing the hostility vector, while
# the disagreement values remain the same.

# Tuples below determine hostility values, and, as a consequence, the strength of the correlation for this experiment.
# There are three values: 
#   1) Strength of the initial correlation: Passed to data.hostility() as the parameter 'a'. Values need to be between 
#      0.0 and 1.0. Values close to 0.0 result in a perfect correlation, while values close to 1.0 equal no correlation.
#   2) Sign of correlation coefficient: Passed to data.hostility() as the parameter 'positive'. 
#      Determines whether the correlation is positive or negative.
#   3) The third value is used to determine if the function data.update_hostility() should be used.
#      This function reduces the strength of the correlation. For instance, the update might change a 
#      disagreement--hostility pair of 2.0--1.99 to 2.0--1.65, and a pair of 1.75--1.73 to 1.75--1.50.
#      This process results in a weaker correlation. Importantly, the within-group and between-group
#      signs stay the same: all hostility values <0.5 stay <0.5, and all hostility values >0.5 remain >0.5.
intervals = ((0.01, False, False), (0.01, False, True), (1.0, True, False), (0.01, True, True), (0.01, True, False))

results = []
for count in range(niter):
    print(count)

    # Create a random network. Generate opinion and disagreement data.
    G = nx.gnm_random_graph(n=nnodes, m=nedges)
    Q = ap._ge_Q(network=G)
    o = data.make_opinion(size=nnodes)
    dis = data.disagreement(opinions=o, network=G)
    
    # Test different hostility values.
    for index, (intv, is_pos, update) in enumerate(intervals):  
        hos = data.hostility(disagree=dis, intv=intv, positive=is_pos)

        if update:
            hos = data.update_hostility(hostile=hos, intv=0.25)
        else:
            pass
    
        # Calculate the measures.
        results.append(data.calculate_measures(index=index, network=G, Q=Q, opinions=o, disagree=dis, hostile=hos))

# Summarize and save results.
columns = ["network", "pearson", "alpha", "avg_likeminded", "avg_crosscutting", "emd_blue", "emd_red", "sai", "pole"]
cis = data.calculate_confidence_intervals(results=results, columns=columns, by="network", niter=niter)
cis.round(2).to_csv("fig_2.csv", index=False)