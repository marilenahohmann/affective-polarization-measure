import torch, torch_geometric
import numpy as np
from scipy.stats import pearsonr, spearmanr

device = "cuda" if torch.cuda.is_available() else "cpu"

def make_tensor(network):
   """
   Convert a NetworkX graph into a PyTorch tensor.

   Parameters:
   network (networkx.Graph): The input network.

   Returns:
   torch_geometric.data.Data: A tensor representation of the network.
    """
   edge_index = [[], []]
   for edge in network.edges:
      edge_index[0].append(edge[0])
      edge_index[1].append(edge[1])
      edge_index[0].append(edge[1])
      edge_index[1].append(edge[0])
   tensor = torch_geometric.data.Data(
      edge_index = torch.tensor(edge_index, dtype = torch.long).to(device)
   )
   return tensor

def _ge_Q(network):
   """
   Compute the Moore-Penrose pseudoinverse of the Laplacian matrix of a network.

   Parameters:
   network (networkx.Graph): The input network.

   Returns:
   numpy.ndarray: The pseudoinverse of the Laplacian matrix of the graph.
   """
   tensor = make_tensor(network)
   L_ei, Lew = torch_geometric.utils.get_laplacian(tensor.edge_index)
   L = torch_geometric.utils.to_dense_adj(edge_index = L_ei, edge_attr = Lew)[0]
   return torch.linalg.pinv(L, hermitian = True).double().cpu().numpy()

def alpha(opinions, hostility, network, Q = None, spearman = True):
   """
   Calculate the affective polarization measure alpha for a given network.

   Parameters:
   opinions (dict): A dictionary containing the opinions of the nodes in the network.
   hostile (dict): A dictionary containing hostility values for each edge in the network.
   network (networkx.Graph): The input network.
   Q (numpy.ndarray, optional): The Moore-Penrose pseudoinverse of the Laplacian matrix of the network
   spearman (bool, optional): If False, uses Pearson correlation instead of Spearman rank correlation.

   Returns:
   float: The affective polarization measure alpha.
   """
   hostility = np.array([hostility[(e[0], e[1])] for e in network.edges])
   disagreement = np.array([abs(opinions[e[0]] - opinions[e[1]]) for e in network.edges])
   if spearman:
      corr = spearmanr(disagreement, hostility)[0]
   else:
      corr = pearsonr(disagreement, hostility)[0]
   if Q is None:
      Q = _ge_Q(network)
   o = np.array([opinions[n] if n in opinions else 0. for n in network.nodes()])
   ge = np.sqrt(o.T.dot(np.array(Q).dot(o)))
   return ge * corr