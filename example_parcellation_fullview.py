import numpy as np
import bct
from brainroisurf.nilearndrawmembership import draw_parcellation_multiview


if __name__ == '__main__':
    template = 'data/templates/atlas/template_638.nii.gz'
    surf_left = 'data/brainmeshes/BrainMesh_ICBM152Left_smoothed.nv.gz'
    surf_right = 'data/brainmeshes/BrainMesh_ICBM152Right_smoothed.nv.gz'
    A = np.loadtxt('data/matrices/Coactivation_matrix_weighted.adj')
    memb,q = bct.community_louvain(A, gamma=1.5) # run community detection on matrix A
    memb = np.asarray(memb) # convert to numpy array
    memb = (memb - memb.min()) + 1  # from 1 to C
    draw_parcellation_multiview(template, surf_left, surf_right, memb, 'coactivation_louvain_fullview.png')