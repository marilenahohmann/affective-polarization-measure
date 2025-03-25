from implementation import affective_polarization as ap  
from implementation import synthetic_experiments as data

# Specifications for the experiment.
clique_size = 25                            # Clique size
niter = 100                                 # Number of repetitions

# This experiment changes the network structure, while the disagreement--hostility correlation remains the same.
# The experiment starts with a network (G0) that consists of two cliques connected by one edge.
# The edges within the cliques are (symmetrically) replaced by edges between the cliques.
# These replacement edges are chose in such a way that the disagreement value of the old (within-clique) edge,
# is almost the same as the new (between-clique) edge.

# The tuple below specifies the number of edges to rewire.
rewire = ((1, 10, 50, 150))

# We use the indices to name the output files.
indices = (3,2,1,0)             

results = []
for count in range(niter):
    print(count)

    # Create the initial network with two cliques. Generate opinion, disagreement, and hostility data.
    G0 = data.make_network(clique_size=clique_size)
    Q0 = ap._ge_Q(G0)
    o = data.make_opinion(size=int(clique_size*2))
    dis = data.disagreement(opinions=o, network=G0)
    hos = data.hostility(disagree=dis, intv=0.25, positive=True)

    # Calculate all measures for the initial network.
    measures0 = data.calculate_measures(index=4, network=G0, Q=Q0, opinions=o, disagree=dis, hostile=hos)

    # The tuple specifies the number of edges to rewire.
    temp = []
    for i, n in enumerate(rewire):
        index = indices[i]
        G1 = data.rewire_network(network=G0, clique_size=clique_size, opinions=o, nrewire=n, margin=0.05)
        Q1 = ap._ge_Q(G1)
        dis1 = data.disagreement(opinions=o, network=G1)
        hos1 = data.hostility(disagree=dis1, intv=0.25, positive=True)
 
        # Only if everything could be calculated successfully, save the results and update the counter.
        measures1 = data.calculate_measures(index=index, network=G1, Q=Q1, opinions=o, disagree=dis1, hostile=hos1)
        temp.append(measures1)

    # Only if everything could be calculated successfully, save the results and update the counter.
    results.append(measures0)
    results.extend(temp) 

# Calculate 95% confidence intervals across the repetitions and save results.
columns = ["network", "pearson", "alpha", "avg_likeminded", "avg_crosscutting", "emd_blue", "emd_red", "sai", "pole"]
cis = data.calculate_confidence_intervals(results=results, columns=columns, by="network", niter=niter)
cis.round(2).to_csv("fig_4.csv", index=False)