# BrainRoiSurf

This simple package contains two modified function from the `nilearn.surface` submodule that help in making beatiful visualization.
You can see nice examples at the page of [my website](carlonicolini.github.io/sections/science/2018/11/27/Plotting-custom-brain-parcellation-beautifully-in-python-with-nilearn.html)


![nilearn roi surface with beatiful colormaps](https://carlonicolini.github.io/static/postfigures/nilearn-brain-parcellation-multiview.jpg)

# Example code
You can generate a picture like the one here, with just a few lines of code.
You just need to clone this repository and run the `example_parcellation_fullview.py`.
It requires the `bct` python package, which can be installed with `pip install bctpy`.

**P.S** data files can only be fetched via the large file extensions of git, called `git lfs`. If you don't know how to install `git lfs`, follow this guide: 

	git lfs clone https://github.com/carlonicolini/brainroisurf
	cd brainroisurf
	python3 example_parcellation_fullview.py

The code can be adapted, this is the basic code to make a beatiful picture.

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

You can provide the brain meshes both as `.nv` file, or as compressed file `.nv.gz`, thanks to `gzip` capabilities of the `load_nv` function.
Same for the nifti files, both `.nii` or `.nii.gz` are supported.

Alternatively you can use the default surfaces provided in `nilearn`.
More to come...


