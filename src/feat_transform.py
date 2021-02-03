""" This module contains functions used for feature engineering steps applied in PathLearn"""

import torch


def feat_transform_func(Vf_slc, Vc_slc, batch_triples, max_paths_num, node_inds, edge_inds, Vf, Vc):
    """
    Feature transformation function for the DBLP dataset used in the demo/paper, that can be used as template/example.\
    It calculates the difference in  year of publication between papers in the paths and papers at the end of each path.\
    The function is used in path_learn.PathL.forward.

    :param Vf_slc: Vf_slc that is constructed in path_learn.PathL.forward.
    :param Vc_slc:  Vc_slc that is constructed in path_learn.PathL.forward.
    :param batch_triples: The batch triples provided in path_learn.PathL.forward.
    :param max_paths_num: Maximum number of paths per pair calculated in path_learn.PathL.forward.
    :param node_inds: Node_inds attribute of PathL instance.
    :param edge_inds: Edge_inds attribute of PathL instance.
    :param Vf: Vf tensor of PathL instance.
    :param Vc: Vc tensor of PathL instance.
    :return: Transformed version of Vf_slc.
    """

    years = torch.zeros(len(batch_triples) * max_paths_num)
    for i, (h, r, t) in enumerate(batch_triples):
       years[i * max_paths_num:i * max_paths_num + max_paths_num] = Vf[node_inds[t], 2] #year of end paper for each pair
    Vf_slc[:, 1, 2] = torch.abs(
       Vf_slc[:, 1, 2] - years)  # Vf_slc[:,1,:] are papers, Vf_slc[:,:,2] are years, Vf_slc[:,1,2] are years of papers
    return Vf_slc
