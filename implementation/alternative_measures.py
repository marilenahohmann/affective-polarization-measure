import numpy as np
import scipy
import networkx as nx
from sklearn.utils import shuffle
from scipy.stats import pearsonr

def dict_to_array(attributes, network):
    """
    Converts a dictionary of values to a numpy array, ordered according to the nodes in the network.

    Parameters:
    attributes (dict): A dictionary specifying values (here: disagreement or hostility) for each edge in the network.
    network (networkx.Graph): The input network.

    Returns:
    numpy.ndarray: An array containing the dictionary values in the same order as the edges appear in the network.

    Note: Every node in the network needs to be present as a key in the attributes dictionary.
    Otherwise, an error will be raised.
    """
    return np.array([attributes[n] for n in network.edges], dtype=float)

def get_signs(hostile):
    """
    Assigns signs to hostility values based on a threshold of 0.5. 
    This function assigns a positive sign (+1) to values less than or equal to 0.5, 
    and a negative sign (-1) to values greater than 0.5.

    Parameters:
    hostile (dict): A dictionary containing hostility values.

    Returns:
    dict: A dictionary where the hostility values are replaced by either +1 or -1.
    """
    return {key:(1 if val <= 0.5 else -1) for key, val in hostile.items()}

# --------------------------------------------------------------------------------------------------------
# Pearson correlation coefficient. 
# --------------------------------------------------------------------------------------------------------

def pearson(x, y, network):
    """
    Calculate the Pearson correlation coefficient between two edge attributes.

    Parameters:
    x (dict): A dictionary containing values of the first attribute (here: disagreement).
    y (dict): A dictionary containing of the second attribute (here: hostility).
    network (networkx.Graph): The input network.

    Returns:
    float: The Pearson correlation coefficient between the two attributes.
    """
    x = dict_to_array(x, network)
    y = dict_to_array(y, network)
    return pearsonr(x, y)[0]

# --------------------------------------------------------------------------------------------------------
# Average Sentiment Measure. 
# --------------------------------------------------------------------------------------------------------

def avg_sentiment(opinions, hostile, network):
    """Calculate the avg hostility for like-minded and cross-cutting edges.

    Parameters:
    opinions (dict): A dictionary containing the opinions of nodes in the network.
    hostile (dict): A dictionary containing the hostility values for each edge in the network.
    network (networkx.Graph): The input network.

    Returns:
    tuple: A tuple containing two values - the average hostility for like-minded (same color) edges and 
           the average hostility for cross-cutting (blue and red) edges.
    """
    likeminded = []
    crosscutting = []

    # Loop over all the edges, and save their hostility values 
    # in 'likeminded' (for edges connecting nodes of same color) or 
    # in 'crosscutting' (for edges connecting a blue and a red node). 
    for edge in network.edges:
        if opinions[edge[0]] <= 0:
            if opinions[edge[1]] <= 0:
                likeminded.append(hostile[edge])         # Both nodes are blue.
            else:
                crosscutting.append(hostile[edge])       # Node 0 is blue, node 1 is red.
        else:
            if opinions[edge[0]] > 0:
                if opinions[edge[1]] > 0:  
                    likeminded.append(hostile[edge])     # Both nodes are red.
                else:
                    crosscutting.append(hostile[edge])   # Node 0 is red, node 1 is blue.
    
    # Calculate the mean hostility value for the like-minded and cross-cutting edges.
    if likeminded:
        likeminded_hos = np.mean(likeminded)
    else:
        likeminded_hos = np.nan

    if crosscutting:
        crosscutting_hos = np.mean(crosscutting)
    else:
        crosscutting_hos = np.nan

    return likeminded_hos, crosscutting_hos

# --------------------------------------------------------------------------------------------------------
# EMD measure by Tyagi et al. (2021). The code below is based on the description of the measure in 
# the paper. Link to paper: https://link.springer.com/article/10.1007/s13278-021-00792-6
# --------------------------------------------------------------------------------------------------------

def in_out_edges(network, group):
    """Determine which edges are in-group and which are out-group based on a given set of edges.

    Parameters:
    network (networkx.Graph): The input network.
    group (list or set): The specified group for which the edges are being evaluated.

    Returns:
    tuple: A tuple containing two lists - the list of edges within the specified group (in-group) 
           and the list of edges connecting the specified group with outside nodes (out-group).
    """
    group_in = []
    group_out = []
    
    for edge in network.edges:
        if edge[0] in group:
            if edge[1] in group:
                group_in.append(edge)    # Both nodes are in the in-group.
            else:
                group_out.append(edge)   # Node 0 is in the in-group, node 1 is in the out-group.
        else:
            if edge[1] in group:
                group_out.append(edge)   # Node 0 is in the out-group, node 1 is in the in-group.
            else:
                pass                     # If node 0 and node 1 are in the out-group, do nothing.

    return group_in, group_out


def ei_index(network, group_edges, hostile):
    """Calculate the E/I index for a given stance group in the network.

    Parameters:
    network (networkx.Graph): The input network.
    group_edges (list or set): The specified group of edges for which the E/I index is being calculated.
    hostile (dict): A dictionary containing the hostility values for each edge in the network.

    Returns:
    float: The calculated E/I index for the specified group, representing the relative balance of positive and negative 
           stances in the in-group compared to the out-group.
    """

    group_in, group_out = in_out_edges(network, group_edges)
    signs = get_signs(hostile=hostile)

    # Get positive edges for in-group and out-group and their hostility values.
    in_pos = sum([hostile[edge] for edge in group_in if signs[edge] == 1])
    in_neg = sum([hostile[edge] for edge in group_in if signs[edge] == -1])

    out_pos = sum([hostile[edge] for edge in group_out if signs[edge] == 1])
    out_neg = sum([hostile[edge] for edge in group_out if signs[edge] == -1])

    # Calculate p^+_{k} and p^-_{k}. Make sure to avoid ZeroDivisionErrors.
    if out_pos == 0 and in_pos == 0:
        p_pos = 0

    else:
        p_pos = (out_pos - in_pos) / (out_pos + in_pos)

    if out_neg == 0 and in_neg == 0:
        p_neg = 0

    else:
        p_neg = (out_neg - in_neg) / (out_neg + in_neg)

    return (p_neg - p_pos) / 2       # Return p_{k}.

def in_out_hostility(network, group, hostile):
    """Get a vector of all in-group and out-group hostility values.

    Parameters:
    network (networkx.Graph): The input network.
    group (list or set): The specified group for which the hostility values are being extracted.
    hostile (dict): A dictionary containing the hostility values for each edge in the network.

    Returns:
    tuple: A tuple containing two numpy arrays - the array of hostility values for in-group edges and 
           the array of hostility values for out-group edges.
    """

    # Get in-group and out-group edges.
    group_in, group_out = in_out_edges(network, group)

    # Get the in-group and out-group hostility values.
    in_hos = np.array([hostile[edge] for edge in group_in])
    out_hos = np.array([hostile[edge] for edge in group_out])

    return in_hos, out_hos

def emd(opinions, hostile, network):
    """
    Calculate the Earth Mover's Distance (EMD)-based affective polarization measure according to Tyagi et al. (2021).

    Parameters:
    opinions (dict): A dictionary containing the opinions of nodes in the network.
    hostile (dict): A dictionary containing the hostility values for each edge in the network.
    network (networkx.Graph): The input network.

    Returns:
    tuple: A tuple containing the affective polarization measures for the 'blue' group and the 'red' group.
    """

    # Split the nodes into a 'blue' group and a 'red' group. We use 0 as the threshold to split the nodes.
    blue = {key:val for key, val in opinions.items() if val <= 0}
    red = {key:val for key, val in opinions.items() if val > 0}

    # For each group, retrieve all in-group and out-group hostility values.
    blue_in, blue_out = in_out_hostility(network, blue, hostile)
    red_in, red_out = in_out_hostility(network, red, hostile)

    # Get valence of the interactions by calculating the E/I index.
    # This index will indicate the sign of the measure: if it is > 0,
    # then the out-group hostility is higher than the in-group hostility.
    ei_blue = ei_index(network, blue, hostile)
    ei_red = ei_index(network, red, hostile)

    # Calculate the affective polarization magnitude as the Earth Mover's Distance.
    blue_ap = scipy.stats.wasserstein_distance(blue_out, blue_in)
    red_ap = scipy.stats.wasserstein_distance(red_out, red_in)

    # Combine the valence and the magnitude for the final affective polarization measure.
    if ei_blue < 0:
        blue_ap = - blue_ap

    if ei_red < 0:
        red_ap = - red_ap

    return blue_ap, red_ap

# --------------------------------------------------------------------------------------------------------
# SAI measure by Fraxanet et al. (2024). The code below is based on the description of the measure in 
# the paper. Link to paper: https://academic.oup.com/pnasnexus/article/3/12/pgae276/7713083
# --------------------------------------------------------------------------------------------------------

def frustration(network, between_edges, hostile):
    """
    Calculate the frustration index for a signed network.

    Parameters:
    network (networkx.Graph): The network for which the frustration index is being calculated.
    between_edges (list): List of edges between two different groups.
    hostile (dict): A dictionary containing hostility values.

    Returns:
    float: The calculated frustration index, normalized by the total number of edges in the network.
    """
    
    hostile = get_signs(hostile)    # Turn hostility values into signed edges (-1/+1).

    pos_between = 0                 # Count the number of positive edges between the communities
    neg_within = 0                  # and negative edges within the communities.
    
    for e in network.edges:
        if e in between_edges:
            if hostile[e] == 1:     # Positive (= non-hostile) interaction
                pos_between += 1
        else:
            if hostile[e] == -1:    # Negative (= hostile) interaction
                neg_within += 1

    return (pos_between + neg_within) / len(network.edges)

def signed_alignment_index(network, opinions, hostile):
    """
    Calculate the signed alignment index according to Fraxanet et al. (2024). 

    Parameters:
    network (networkx.Graph): The network for which the signed alignment index is being calculated.
    opinions (dict): A dictionary containing the opinions of nodes in the network.
    hostile (dict): A dictionary mapping each edge to its corresponding hostility value.

    Returns:
    float: The signed alignment index.
    """
    
    # We consider two groups of nodes: blue nodes (opinions < 0) and red nodes (opinions > 0).
    # To find edges within and between groups, we first split the nodes into a blue and red group,
    # and then retrieve the set of all edges between the red nodes and blue nodes.
    blue = set([key for key, value in opinions.items() if value <= 0])
    _, between_edges = in_out_edges(network=network, group=blue)

    # We first calculate the frustration index for the network (l_star). To normalize the measure,
    # we use the mean frustration index of a null model. In the null model, the signs (hostility values) 
    # are randomly shuffled, while the network structure and the node partition into a 
    # blue group and a red group stay fixed.
    l_star = frustration(network=network, between_edges=between_edges, hostile=hostile)
    shuff = hostile.copy()
    l_tilde = np.mean([frustration(network=network, 
                                   between_edges=between_edges, 
                                   hostile=dict(zip(hostile.keys(), shuffle(list(shuff.values()))))) 
                                   for _ in range(10)])
    return (1 - (l_star / l_tilde))

# --------------------------------------------------------------------------------------------------------
# POLE polarization measure by Huang et al. (2022). Link to code repo: https://github.com/zexihuang/POLE
# Link to paper: https://dl.acm.org/doi/abs/10.1145/3488560.3498454
# --------------------------------------------------------------------------------------------------------

def signed_adjacency_matrix(G):
    """
    Adjacency matrix for the signed graph with positive/negative edge weights.

    Parameters:
    G (networkx.Graph): Input graph.

    Returns:
    numpy.ndarray: Signed adjacency matrix.
    """
    return nx.adjacency_matrix(G, sorted(G.nodes), weight='weight').toarray()


def unsigned_adjacency_matrix(G):
    """
    Adjacency matrix for the unsigned graph with absolute edge weights.

    Parameters:
    G (networkx.Graph): Input graph.

    Returns:
    numpy.ndarray: Unsigned adjacency matrix.
    """
    return np.abs(signed_adjacency_matrix(G))


def unsigned_degree_vector(G):
    """
    Degree vector for the unsigned graph.

    Parameters:
    G (networkx.Graph): Input graph.

    Returns:
    numpy.ndarray: Unsigned degree vector.
    """
    A_abs = unsigned_adjacency_matrix(G)
    return A_abs.sum(axis=1)


def unsigned_random_walk_stationary_distribution_vector(G):
    """
    Stationary distribution vector for unsigned random-walk dynamics.
    pi = d/vol(G).

    Parameters:
    G (networkx.Graph): Input graph.

    Returns:
    numpy.ndarray: Unsigned random-walk stationary distribution vector.
    """
    d_abs = unsigned_degree_vector(G)
    return d_abs/np.sum(d_abs)


def unsigned_random_walk_stationary_distribution_matrix(G):
    """
    Stationary distribution matrix for unsigned random-walk dynamics.
    Pi = diag(pi).

    Parameters:
    G (networkx.Graph): Input graph.

    Returns:
    numpy.ndarray: Unsigned random-walk stationary distribution matrix.
    """
    return np.diag(unsigned_random_walk_stationary_distribution_vector(G))


def signed_random_walk_laplacian_matrix(G):
    """
    Random-walk Laplacian matrix for the signed graph.
    L_rw = I - D^-1 A.

    Parameters:
    G (networkx.Graph): Input graph.

    Returns:
    numpy.ndarray: Signed random-walk Laplacian matrix.
    """
    D_1 = np.diag(1/unsigned_degree_vector(G))
    A = signed_adjacency_matrix(G)
    return np.identity(G.number_of_nodes()) - D_1 @ A


def unsigned_random_walk_laplacian_matrix(G):
    """
    Random-walk Laplacian matrix for the unsigned graph.
    L_rw_abs = I - D^-1 A_abs.

    Parameters:
    G (networkx.Graph): Input graph.

    Returns:
    numpy.ndarray: Unsigned random-walk Laplacian matrix.
    """
    D_1 = np.diag(1/unsigned_degree_vector(G))
    A_abs = unsigned_adjacency_matrix(G)
    return np.identity(G.number_of_nodes()) - D_1 @ A_abs


def transition_matrix(L, t):
    """
    Transition matrix based on the Laplacian matrix and Markov time.
    M(t) = exp(-Lt).

    Parameters:
    L (numpy.ndarray): Laplacian matrix.
    t (float): Markov time.

    Returns:
    scipy.sparse.csc_matrix: Transition matrix.
    """
    return scipy.sparse.linalg.expm(- L * t)


def dynamic_similarity_matrix(M_t, W):
    """
    Dynamic similarity matrix.
    R(t) = M(t)^T W M(t).

    Parameters:
    M_t (scipy.sparse.csc_matrix): Transition matrix.
    W (numpy.ndarray): Weight matrix.

    Returns:
    numpy.ndarray: Dynamic similarity matrix.
    """
    return M_t.T @ W @ M_t


def signed_autocovariance_matrix(G, t):
    """
    Signed autocovariance matrix based on signed transition matrix and unsigned stationary distributions.
    R = M(t)^T (Pi - pi pi^T) M(t).

    Parameters:
    G (networkx.Graph): Input graph.
    t (float): Markov time.

    Returns:
    numpy.ndarray: Signed autocovariance similarity matrix.
    """

    pi = unsigned_random_walk_stationary_distribution_vector(G)
    Pi = unsigned_random_walk_stationary_distribution_matrix(G)
    W = Pi - np.outer(pi, pi)

    L_rw = signed_random_walk_laplacian_matrix(G)
    M_t = transition_matrix(L_rw, t)

    return dynamic_similarity_matrix(M_t, W)


def unsigned_autocovariance_matrix(G, t):
    """
    Signed autocovariance matrix based on unsigned transition matrix and unsigned stationary distributions.
    R(t)_abs = M(t)_abs^T (Pi - pi pi^T) M(t)_abs.

    Parameters:
    G (networkx.Graph): Input graph.
    t (float): Markov time.

    Returns:
    numpy.ndarray: Unsigned autocovariance similarity matrix.
    """

    pi = unsigned_random_walk_stationary_distribution_vector(G)
    Pi = unsigned_random_walk_stationary_distribution_matrix(G)
    W = Pi - np.outer(pi, pi)

    L_rw = unsigned_random_walk_laplacian_matrix(G)
    M_t = transition_matrix(L_rw, t)

    return dynamic_similarity_matrix(M_t, W)

def add_signs(network, hostile):
    """
    Turn hostility values into edge signs, and add them as edge attributes with the name "weight".

    Parameters:
    network (networkx.Graph): The network to which signs are added.
    hostile (dict): A dictionary containing the hostility values for the edges.

    Returns:
    networkx.Graph: The network with signs added to the edges.
    """
    hostile = get_signs(hostile)
    nx.set_edge_attributes(G=network, values=hostile, name="weight")
    return network

def pole(network, hostile, t=1.0):
    """
    Compute the POLE polarization measure.

    Parameters:
    network (networkx.Graph): Input graph.
    t (float): Markov time. Default is 1.0.

    Returns:
    float: Returns the network-level polarization score (mean of all node-level polarization scores).
    """
    t = 10 ** t
    network = add_signs(network=network, hostile=hostile)
    M = transition_matrix(signed_random_walk_laplacian_matrix(network), t)
    M_abs = transition_matrix(unsigned_random_walk_laplacian_matrix(network), t)

    node_scores = np.asarray([scipy.stats.pearsonr(signed, unsigned)[0] for signed, unsigned in zip(M.T, M_abs.T)])
    node_scores = np.nan_to_num(node_scores, nan=0.0)  # Replace Nan with 0.0.

    return node_scores.mean()
