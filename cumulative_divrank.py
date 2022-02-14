import numpy as np
import pandas as pd
import networkx as nx
from networkx.exception import NetworkXError


def cum_divrank(G, alpha=0.25, d=0.85, personalization=None,
            max_iter=100, tol=1.0e-6, nstart=None, weight='weight',
            dangling=None):
    '''
    Returns the DivRank (Diverse Rank) of the nodes in the graph.
    This code is based on networkx.pagerank.
    
    Parameters
    ----------
    G : graph
      A NetworkX graph.  Undirected graphs will be converted to a directed
      graph with two directed edges for each undirected edge.

    alpha : controls strength of self-link [0.0-1.0]

    d: the damping factor, default=0.85

    personalization: personalization vector and it should be dict, optional
    
    max_iter : integer, optional
      Maximum number of iterations in power method eigenvalue solver.

    tol : float, optional
      Error tolerance used to check convergence in power method solver.

    nstart : dictionary, optional
      Starting value of PageRank iteration for each node.

    weight : key, optional
      Edge data key to use as weight.  If None weights are set to 1.

    dangling: dict, optional
      The outedges to be assigned to any "dangling" nodes, i.e., nodes without
      any outedges. The dict key is the node the outedge points to and the dict
      value is the weight of that outedge. By default, dangling nodes are given
      outedges according to the personalization vector (uniform if not
      specified). This must be selected to result in an irreducible transition
      matrix (see notes under google_matrix). It may be common to have the
      dangling dict to be the same as the personalization dict.


    Returns
    -------
    cumulative divrank : dictionary
       Dictionary of nodes with Diverse Rank as value
    Reference:
      Qiaozhu Mei and Jian Guo and Dragomir Radev,
      DivRank: the Interplay of Prestige and Diversity in Information Networks,
      http://www-personal.umich.edu/~qmei/pub/kdd10-divrank.pdf
    '''

    if len(G) == 0:
        return {}

    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G
    
    print("original graph:")
    print(nx.to_numpy_array(D))
    print("\n")

    # Create a copy in (right) stochastic form
    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()
    
    print("weighted graph:")
    print(nx.to_numpy_array(W))
    print("\n")

    # self-link (DivRank)
    for n in W.nodes():
        for n_ in W.nodes():
            if n != n_ :
                if n_ in W[n]:
                    W[n][n_]['weight'] *= alpha
            else:
                if n_ not in W[n]:
                    W.add_edge(n, n_)
                W[n][n_]['weight'] = 1.0 - alpha
    

    print("weighted graph with self-link:")
    print(nx.to_numpy_array(W))
    print("\n")

    # Choose fixed starting vector if not given
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        # Normalized nstart vector
        s = float(sum(nstart.values()))
        x = dict((k, v / s) for k, v in nstart.items())
    
    if personalization is None:
        # Assign uniform personalization vector if not given
        p = dict.fromkeys(W, 1.0 / N)
    else:
        missing = set(G) - set(personalization)
        if missing:
            raise NetworkXError('Personalization dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        s = float(sum(personalization.values()))
        p = dict((k, v / s) for k, v in personalization.items())

    if dangling is None:
        # Use personalization vector if dangling vector not specified
        dangling_weights = p
    else:
        missing = set(G) - set(dangling)
        if missing:
            raise NetworkXError('Dangling node dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        s = float(sum(dangling.values()))
        dangling_weights = dict((k, v/s) for k, v in dangling.items())
    
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]
    print("list of dangling nodes: ", dangling_nodes)
    print("\n")

    x_initial = x.copy()
    x_cumm = x_initial
    for _ in range(max_iter):
        x_prev = x
        for to in set(W):
            first_part = (1-d)*p[to]
            inner_sum = 0
            for frm in set(W):
                try:
                    neum = W[frm][to]['weight'] * x_cumm[to]
                except:
                    neum = 0
                D_t = sum([W[frm][new_to]['weight'] * x_cumm[new_to] for new_to in W[frm]])
                denom = D_t
                
                dangling_node_pgrnk = x_prev[frm] if frm in dangling_nodes else 0
                dangling_node_edgwt = dangling_weights[frm]
                
                inner_sum += (neum/denom)*x[frm] + (dangling_node_edgwt*dangling_node_pgrnk)
            second_part = d*inner_sum
            total_part = first_part + second_part
            x[to] = total_part
        x_s = float(sum(x.values()))
        x = dict((k, v / x_s) for k, v in x.items())
        
        for cumm_to in set(W):
            x_cumm[cumm_to] += x[cumm_to]
        
        err = sum([abs(x[n] - x_prev[n]) for n in x])
        if err < N*tol:
            print("sum of cumulative divrank: ", sum(x.values()))
            print("\n")
            print("cumulative divrank:")
            return(x)
            break
            
    
    raise NetworkXError('cumm_divrank: power iteration failed to converge '
                        'in %d iterations.' % max_iter)


