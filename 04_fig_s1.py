import networkx as nx
from implementation import affective_polarization as ap
from implementation import synthetic_experiments as data

# Specifications for the experiment.
nnodes = 50                                 # Network size
nedges = 610                                # Number of edges
niter = 100                                 # Number of repetitions

# In this experiment, we change change the opinion distribution by changing the upper bound of the interval 
# from which the opinions are drawn. The disagreement--hostility relation remains the same.

# The tuple below determines the upper bound of the interval from which the opinions are drawn.
# The lower this value, the lower the level of disagreement in the random network we consider below.
stops = (0.2, 0.4, 0.6, 0.8, 1.0)                   

results = []
for count in range(niter):

    print(count)

    # Create the network.
    G = nx.gnm_random_graph(n=nnodes, m=nedges)
    Q = ap._ge_Q(network=G)
    o = data.make_clipped_opinion_distribution(size=nnodes, stop=stops[-1])
    dis = data.disagreement(o, G)
    hos = data.hostility(disagree=dis, intv=0.25, positive=True)

    # Test different opinion values (= different disagreement distributions).
    for index, stop in enumerate(stops):
        o = data.make_clipped_opinion_distribution(size=nnodes, stop=stop)
        d = data.disagreement(o, G)
        results.append(data.calculate_measures(index=index, network=G, Q=Q, opinions=o, disagree=d, hostile=hos))
           
# Summarize and save results.

# Note: When calculating the SAI measure, the confidence interval returned here is (NaN, NaN).
# This is probably due to the fact that all SAI values are identical.
# As the table shows, the SAI mean is 1.0, and the SEM is 0.0 for across all examples (a) to (e).
# We, therefore, report the confidence interval (1.0, 1.0) in the paper.

columns = ["network", "pearson", "alpha", "avg_likeminded", "avg_crosscutting", "emd_blue", "emd_red", "sai", "pole"]
cis = data.calculate_confidence_intervals(results=results, columns=columns, by="network", niter=niter)
cis.to_csv("fig_s1.csv", index=False)