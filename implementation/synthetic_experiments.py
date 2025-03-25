import numpy as np
import pandas as pd
import scipy
import networkx as nx
import random
from implementation import affective_polarization as corr  
from implementation import alternative_measures as alt

def replace_opinions(opinions, i, val):
    """
    Replaces a specified number of opinions (up to the i'th opinions) with a specified value.

    Parameters:
    opinions (numpy array): Values representing individual opinions.
    i (int): The number of opinions to replace.
    val (float): The value to replace the selected opinions with.

    Returns:
    numpy array: A new array containing the modified opinions.
    """
    opinions[:i] = val
    return opinions

def make_opinion(size, i=0, val=None):
    """
    Generates a dictionary with node opinions.

    Parameters:
    size (int): Number of nodes in the network.
    i (int): The number of opinions to replace.
    val (float): The value to replace the selected opinions with.

    Returns:
    dict: An opinion dictionary with nodes (keys) and opinions (values).
    """

    o = np.linspace(start=0.01, stop=1.0, num=size//2)
    o = replace_opinions(opinions=o, i=i, val=val)
    o = np.concatenate((o, -o))
    return {i: o[i] for i in range(len(o))}

def make_clipped_opinion_distribution(size, stop):
    """
    Generates equispaced opinion values in [stop-0.1, stop], and returns a dictionary.
    This function is used to generate figure S1 in the Supplementary Materials.    

    Parameters:
    size (int): Number of nodes in the network.
    stop (int): The upper bound of the interval.

    Returns:
    dict: An opinion dictionary with nodes (keys) and opinions (values).
    """

    o = np.linspace(start=stop-0.1, stop=stop, num=size//2)
    o = np.concatenate((o, -o))
    return {i: o[i] for i in range(len(o))}

def disagreement(opinions, network):
    """
    Calculate the absolute opinion difference (= disagreement) between connected nodes in the network.

    Parameters:
    opinions (dict): A dictionary containing the opinions of the nodes in the network.
    network (networkx.Graph): The input network.

    Returns:
    dict: A dictionary with edges (= node pair) as keys and their disagreement as values.
    """
    return {tuple(sorted(edge)):(abs(opinions[edge[0]] - opinions[edge[1]])) for edge in network.edges}

def lower(val, interval):
    """
    Calculate lower bound based on val (the value) and the interval given.

    Parameters:
    val (int or float): The value for which the lower bound is being calculated.
    interval (int or float): The interval used for the calculation.

    Returns:
    float: The lower bound.
    """
    return 0 if (val/2)-interval < 0 else (val/2)-interval

def upper(val, interval):
    """
    Calculate upper bound based on val (the value) and the interval given.
    
    Parameters:
    val (int or float): The value for which the upper bound is being calculated.
    interval (int or float): The interval used for the calculation.

    Returns:
    float: The upper bound.
    """
    return 1 if (val/2)+interval > 1 else (val/2)+interval

def hostility(disagree, intv, positive):
    """
    Generate a dictionary of hostility values.

    Parameters:
    disagree (dict): A dictionary containing the disagreement values for each edge in the network.
    intv (float): The interval used for generating the hostility values.
    positive (bool): If True, the values are drawn from an interval surrounding the respective values.
                     If False, the values are drawn from the opposite end of the hostility spectrum.

    Returns:
    dict: A dictionary with edges as keys and hostility as values.

    Note:
    - intv == 0: max correlation, the disagreement and hostility values are the same
    - intv == 1: no correlation, the hostility values are randomly generated
    """
    hostile = {}
    for key, val in disagree.items():
        if positive is True:
            v = np.random.uniform(low=lower(val, intv), high=upper(val, intv))
        else:
            v = np.random.uniform(low=1-upper(val, intv), high=1-lower(val, intv))
        if isinstance(key, tuple):
            k = tuple(sorted(key))
        else:
            k = key
        hostile[k] = v
    return hostile

def update_hostility(hostile, intv):
    """
    Update the hostility levels to lower the strength of the disagreement/hostility correlation.

    Parameters:
    hostile (dict): A dictionary containing hostility values for each edge in the network.
    intv (float): The interval within which the hostility values will be adjusted. Should be a non-negative value.

    Returns:
    dict: A dictionary containing the updated hostility values.

    Note:
    The function maintains the hostility threshold used for the SAI and POLE.
    I.e., all non-hostile (<0.5) values stay in [0, 0.5], and all hostile (>0.5) values in [0.5, 1.0].
    We use this to show that some of the alternative measures cannot capture nuances in the hostility
    distribution.
    """
    return {key: np.random.uniform(np.clip(val-intv, 0.5, 1.0), np.clip(val+intv, 0.5, 1.0)) if val > 0.5 else np.random.uniform(np.clip(val-intv, 0.0, 0.5), np.clip(val+intv, 0.0, 0.5)) for key, val in hostile.items()}
    
def make_network(clique_size):
    """
    Create a network with two cliques connected by one edge.

    Parameters:
    clique_size (int): The size of the cliques in the network.

    Returns:
    networkx.Graph: The network.

    """
    G = nx.disjoint_union(*[nx.complete_graph(clique_size).copy()]*2)
    G.add_edge(0, clique_size)
    return G

def update_network(network, old_edge, new_edge):
    """
    Update the network by replacing an edge within one of the cliques by an edge between the cliques,
    while maintaining a similar disagreement value.

    Parameters:
    network (networkx.Graph): The network to be modified.
    old_edge (tuple): The tuple representing the old edge to be replaced.
    new_edge (tuple): The tuple representing the new edge to be created.

    Returns:
    bool: True if the update is successful, False if the new edge already exists in the network.
    """
    new_edge = tuple(sorted(new_edge))  
    old_edge = tuple(sorted(old_edge))

    if new_edge in network.edges:
        return False
    else:
        network.remove_edge(old_edge[0], old_edge[1])
        network.add_edge(new_edge[0], new_edge[1])
        return True
    
def rewire_network(network, clique_size, opinions, nrewire, margin):
    """
    Rewire the network while keeping the disagreement distribution the same (+/- a small error margin).

    Parameters:
    network (networkx.Graph): The network to be rewired.
    clique_size (int): The size of the cliques in the network.
    opinions (dict): A dictionary containing opinion values for each node in the network.
    nrewire (int): The number of rewiring operations to be performed.
    margin (float): The error margin for maintaining the disagreement value during rewiring.

    Returns:
    networkx.Graph: The rewired network based on the specified criteria.
    """

    g = network.copy()
    cliques = ((0, clique_size), (clique_size, 2*clique_size))
    n = 0

    # Continue looping until the specified number of edges have been rewired.
    while n <= nrewire:
        for i, clique in enumerate(cliques):

            # Select a within-clique edge (node1, node2), and a node from the other clique (node3).
            # Then calculate the disagreement value for the within-clique edge.
            node1, node2 = random.choice([e for e in g.edges if e[0] in range(*clique) if e[1] in range(*clique)])
            node3 = random.choice(range(*cliques[(i + 1) % len(cliques)]))
            d = abs(opinions[node1] - opinions[node2])

            # If the disagreement values for the pair node1--node3, or node2--node3 is similar to the
            # old disagreement value (+/- a small error margin), remove the old edge and create the new edge.
            # If not, continue the loop.
            if d-margin < abs(opinions[node1] - opinions[node3]) < d+margin:
                if update_network(network=g, old_edge=(node1, node2), new_edge=(node1, node3)):
                    n += 1
                else:
                    pass

            elif d-margin < abs(opinions[node2] - opinions[node3]) < d+margin:
                if update_network(network=g, old_edge=(node1, node2), new_edge=(node2, node3)):
                    n += 1
                else:
                    pass
            else:
                pass
    return g


def calculate_measures(index, network, Q, opinions, disagree, hostile):
    """
    Calculate the various affective polarization measures tested in the experiments:

    Parameters:
    index (int): Current iteration of the experiment.
    network (networkx.Graph): The input network.
    Q (float): Pseudoinverse of the Laplacian matrix of the network.
    opinions (dict): A dictionary containing the opinions of the nodes in the network.
    disagree (dict): A dictionary containing the disagreement values for each edge in the network.
    hostile (dict): A dictionary containing hostility values for each edge in the network.

    Returns:
    tuple: A tuple containing the index, Pearson correlation, our affective polarization measure alpha, 
    average sentiment (like-minded), average sentiment (cross-cutting), 
    Earth Mover's Distance measure (blue group), Earth Mover's Distance measure (red group),
    and the Signed Alignment Index, and POLE.
    """
    pearson = alt.pearson(x=disagree, y=hostile, network=network)    
    alpha = corr.alpha(opinions=opinions, hostility=hostile, network=network, Q=Q)
    avg_likeminded, avg_crosscutting = alt.avg_sentiment(opinions=opinions, hostile=hostile, network=network)
    emd_blue, emd_red = alt.emd(opinions=opinions, hostile=hostile, network=network)
    sai = alt.signed_alignment_index(network=network, opinions=opinions, hostile=hostile)
    pole = alt.pole(network=network, hostile=hostile)
    return index, pearson, alpha, avg_likeminded, avg_crosscutting, emd_blue, emd_red, sai, pole

def calculate_confidence_intervals(results, columns, by, niter):
    """
    Calculate confidence intervals (based on the t-distribution) across several iterations of an experiment.

    Parameters:
    results (list of tuples): The results for which confidence intervals need to be calculated.
    columns (list): A list of column names for the results.
    by (str): The column name used for grouping the results and calculating confidence intervals.
    niter (int): The number of iterations used in the calculations. Should be a positive integer.

    Returns:
    pandas DataFrame: A DataFrame containing the mean, standard error, and confidence intervals.
    """
    df = pd.DataFrame(data=results, columns=columns)
    cis = df.groupby(by=by).agg(["mean", scipy.stats.sem])
    cis.columns = ['_'.join(col).strip() for col in cis.columns.values]
    for col in columns[1:]:
        cis[f"{col}_ci"] = cis.apply(lambda x: scipy.stats.t.interval(confidence=0.95, df=niter-1, loc = x[f"{col}_mean"], scale = x[f"{col}_sem"]), axis = 1).apply(lambda x: np.round(x,2))
    return cis
