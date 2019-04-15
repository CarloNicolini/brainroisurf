import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn import surface
from nilearn.image import load_img, smooth_img
from .surf_plotting import plot_surf_roi

def normalize_v3(arr):
    """ Normalize a numpy array of 3 component vectors shape=(n,3) """
    lens = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)

    # hack
    lens[lens == 0.0] = 1.0

    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normals(vertices, triangles):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[triangles]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[triangles[:, 0]] += n
    norm[triangles[:, 1]] += n
    norm[triangles[:, 2]] += n
    normalize_v3(norm)

    return norm

def load_nv(filename):
    '''
    Load files in the .nv format as the Surf data in the BrainetViewer
    First line is a comment and is skipped
    '''
    if os.path.splitext(filename)[1]=='.gz':
        import gzip
        myopen = lambda x: gzip.open(x,'rt')
    else:
        myopen = lambda x: open(x)
    
    import itertools
    num_vertices = int(myopen(filename).readlines()[1])
    num_faces = int(myopen(filename).readlines()[2+num_vertices])
    XYZ, faces = None, None
    with myopen(filename) as f_input:
        XYZ = np.loadtxt(itertools.islice(f_input, 0, num_vertices+2),
                         delimiter=' ', skiprows=2, dtype=np.float32)
    with myopen(filename) as f_input:
        faces = np.loadtxt(itertools.islice(f_input, num_vertices+3,
                                            num_vertices+num_faces+3),
                            delimiter=' ', skiprows=0, dtype=np.int32)
    return XYZ, faces - 1

def create_mpl_integer_cmap(name, num_classes, add_gray=True):
    import brewer2mpl
    from matplotlib import colors
    bmap = brewer2mpl.get_map(name, 'qualitative', num_classes)
    from copy import copy
    cols = copy(bmap.mpl_colors)
    if add_gray:
        cols.insert(0,[0,0,0,0]) # add black for the unmapped areas
    cmap = colors.LinearSegmentedColormap.from_list(name, cols)
    return cmap

def membership_to_rois(template, mesh, memb, **kwargs):
    '''
    Input:
        template: a nibabel.nifti1.Nifti1Image object
        mesh: a string of a mesh, or a list of two np.array with vert and faces
        memb: a np.array of integers of the parcel membership

    '''
    from copy import copy
    template2 = copy(template)
    radius = kwargs.get('radius', 0.01)
    # Put the specific node module index inside the atlas nifti, thus changing 
    # the volume content
    indices = []
    all_parcels = np.unique(template2.get_fdata().flatten()).astype(np.int32)
    # find the indices of the voxels of a given parcel
    # since it's sorted we will exclude the 0 parcel
    for parcel in all_parcels[1:]: # avoid parcel 0 which is empty space
        i, j, k = np.where(template2.get_fdata(caching='fill') == parcel)
        indices.append((i, j, k))

    # # Put the membership as from memb in those voxels
    for parcel in all_parcels[1:]:
        # set the value to the voxesl as the parcel membership
        i,j,k = indices[parcel-1][0], indices[parcel-1][1], indices[parcel-1][2]
        template2.get_data(caching='fill')[i,j,k] = memb[parcel-1]
            
    from nilearn.surface import vol_to_surf
    memb_rois = surface.vol_to_surf(template2, mesh,
                                    interpolation='nearest',
                                    radius=radius)
    return memb_rois


def draw_community_fsaverage():
    
    fsaverage = datasets.fetch_surf_fsaverage5()
    template = load_img('/home/carlo/workspace/communityalg/data/template_638.nii')
    memb = np.loadtxt('Memb1.txt')
    plot_parcellation(template, fsaverage['infl_left'], memb, bg_data=None, output_file = 'surf_inflated.png')


def plot_parcellation(template, surf_mesh, membership, **kwargs):
    '''
    Input:
        template: a nibabel.nifti1.Nifti1Image object
        mesh: a string of a mesh, or a list of two np.array with vert and faces
        memb: a np.array of integers of the parcel membership

    kwargs:
        hemi : {'left', 'right'}, default is 'left'. Inherited from plot_surf_roi.
            Hemispere to display.

    view: {'lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior'}, default is 'lateral'.
        Otherwise specify the view angle as a tuple (elevation, azimuth)
        View of the surface that is rendered.
    
    cmap : matplotlib colormap in str or colormap object, default 'Set3'
        To use for plotting of the stat_map. Either a string
        which is a name of a matplotlib colormap, or a matplotlib
        colormap object. The colormap is passed through brewer2mpl package to make discrete colormaps.
    alpha : float, alpha level of the mesh (not the stat_map), default 'auto'
        If 'auto' is chosen, alpha will default to .5 when no bg_map is
        passed and to 1 if a bg_map is passed.

    bg_on_data : bool, default is False
        If True, and a bg_map is specified, the stat_map data is multiplied
        by the background image, so that e.g. sulcal depth is visible beneath
        the stat_map.
        NOTE: that this non-uniformly changes the stat_map values according
        to e.g the sulcal depth.

    darkness: float, between 0 and 1, default 1
        Specifying the darkness of the background image. 1 indicates that the
        original values of the background are used. .5 indicates the
        background values are reduced by half before being applied.

    title : str, optional
        Figure title.

    output_file: str, or None, optional
        The name of an image file to export plot to. Valid extensions
        are .png, .pdf, .svg. If output_file is not None, the plot
        is saved to a file, and the display is closed.

    axes: instance of matplotlib axes, None, optional
        The axes instance to plot to. The projection must be '3d' (e.g.,
        `figure, axes = plt.subplots(subplot_kw={'projection': '3d'})`,
        where axes should be passed.).
        If None, a new axes is created.

    figure: instance of matplotlib figure, None, optional
        The figure instance to plot to. If None, a new figure is created.

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage5 : For surface data object to be
        used as background map for this plotting function.

    nilearn.plotting.plot_surf : For brain surface visualization.

    '''

    surf_data = None
    if isinstance(surf_mesh, str):
        import os
        filename, file_extension = os.path.splitext(surf_mesh)
        # check if it is a '.nv.gz' or a '.nv' file, in case load it with the load_nv method
        is_nv_file = file_extension=='.nv' or (file_extension=='.gz' and os.path.splitext(filename)[1]=='.nv')
        if is_nv_file:
            surf_data = list(load_nv(surf_mesh))
        else:
            from nilearn.surface import load_surf_mesh
            surf_data = load_surf_mesh(surf_mesh)
    elif isinstance(surf_mesh, list):  # user provided a list of vertices and faces
        if isinstance(surf_mesh[0], np.ndarray) and isinstance(surf_mesh[1], np.ndarray):
            surf_data = surf_mesh
        else:
            raise ValueError('Must provide a list of two numpy arrays with \
                             vertices and faces')
    else:
        raise ValueError('Must provide either a string with the .gii filename \
                         or a list of two numpy arrays with vertices and \
                         faces of the mesh')

    hemi = kwargs.get('hemi', None)
    elev, azim = kwargs.get('view', (45,45))
    darkness = kwargs.get('darkness', 1.5)
    alpha = kwargs.get('alpha', 'auto')
    num_modules = int(np.max(membership))  # use the maximum community number, to avoid 0 on empty nifti voxels
    cmap = kwargs.get('cmap', create_mpl_integer_cmap('Set3', num_modules))
    vmin = np.min(membership)
    vmax = np.max(membership)
    bg_data = kwargs.get('bg_map', None)
    if bg_data is None:  # compute normals and lightning directions
        normals = compute_normals(*surf_data)
        l = np.max(surf_data[0]) # position of light as a scalar
        light_pos = kwargs.get('light_pos', np.array([0, 0, 4*l]))
        light_dir = surf_data[0] - light_pos  # use broadcasting, vertices - light
        light_intensity = np.einsum('ij,ij->i', light_dir, normals)
        bg_data = light_intensity**kwargs.get('shininess',2)
    bg_on_data = kwargs.get('bg_on_data', True)

    # Compute the colors of the mesh vertices based on nodal membership
    mesh_vert_memb = membership_to_rois(template, surf_data , membership)

    #from nilearn import plotting
    #plotting.plot_roi(template, colorbar=True, cmap=kwargs.get('cmap', create_mpl_integer_cmap('Set3', num_modules, False)))
    #plt.savefig('vol.png')
    fig = plot_surf_roi(surf_data,
                        roi_map=mesh_vert_memb,
                        hemi=hemi,
                        view=(elev, azim),
                        vmin=vmin,
                        vmax=vmax,
                        avg_method='max',
                        darkness=darkness,
                        alpha=alpha,
                        cmap=cmap,
                        bg_map=bg_data,
                        bg_on_data=bg_on_data,
                        colorbar=kwargs.get('colorbar',True),
                        num_modules=num_modules,
                        output_file=kwargs.get('output_file',None),
                        dpi=kwargs.get('dpi',200),
                        axes=kwargs.get('axes',None))
    return fig, mesh_vert_memb



def draw_parcellation_multiview(template_file, surface_left_file, surface_right_file, memb, output_file=None, **kwargs):
    template = load_img(template_file)
    #template = load_img('L_06P_s0_Newman.nii')
    #surf_mesh_left =  '/home/carlo/workspace/Brainet2017/Data/SurfTemplate/BrainMesh_ICBM152Left_smoothed.nv'
    #surf_mesh_right = '/home/carlo/workspace/Brainet2017/Data/SurfTemplate/BrainMesh_ICBM152Right_smoothed.nv'
    #plot_parcellation(template, surf_mesh, memb, bg_data=None, view=(45,45), colorbar=True, output_file = 'surf.png')
    num_modules = int(np.max(memb))
    cmap = create_mpl_integer_cmap('Set3', num_modules)

    # Create the multiple view figure
    fig, ax = plt.subplots(ncols=2, nrows=2, subplot_kw={'projection': '3d'})
    fig.text(0.1, 0.85, 'L', fontsize=16)
    fig.text(0.9, 0.85, 'R', fontsize=16)
    _, colors = \
    plot_parcellation(template, surface_left_file,  memb, view=(0,180), cmap=cmap,  axes=ax[0,0], colorbar=False)
    plot_parcellation(template, surface_right_file, memb, view=(0,0),   cmap=cmap,  axes=ax[0,1], colorbar=False)
    plot_parcellation(template, surface_left_file,  memb, view=(0,0),   cmap=cmap,  axes=ax[1,0], colorbar=False)
    plot_parcellation(template, surface_right_file, memb, view=(0,180), cmap=cmap,  axes=ax[1,1], colorbar=False)

    for i in range(2):
        for j in range(2):
            ax[i,j].dist = kwargs.get('zdist',6)

    # add an axis for the colorbar in the center
    cax = fig.add_axes([0.5, 0.325, 0.025, 0.25])
    # add the colorbar to the whole figure
    import matplotlib.cm as cm
    mappable = cm.ScalarMappable(cmap= cmap)
    mappable.set_array(colors)
    cbar = fig.colorbar(mappable, cax=cax, shrink=0.75, aspect=5,
                        boundaries = range(0, num_modules + 2),
                        values = range(0, num_modules + 2))
    cbar.set_ticks( np.array(range(0, num_modules + 1)) + 0.5 )
    cbar.set_ticklabels( [' '] + list(range(1, num_modules + 1)) ) # altezza dei tick labels OK
    
    #for i in range(len(ax)):
    #    ax[i].dist = kwargs.get('zdist',4)

    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.0, hspace=0.0)
    if output_file is None:
        return fig
    else:
        fig.savefig(output_file, dpi=kwargs.get('dpi',200), bbox_inches='tight', pad_inches=0)

# if __name__ == '__main__':
#     template = '/home/carlo/workspace/communityalg/data/template_638.nii'
#     surf_left = '/home/carlo/workspace/Brainet2017/Data/SurfTemplate/BrainMesh_ICBM152Left_smoothed.nv'
#     surf_right = '/home/carlo/workspace/Brainet2017/Data/SurfTemplate/BrainMesh_ICBM152Right_smoothed.nv'
#     A = np.loadtxt('/home/carlo/workspace/communityalg/data/Coactivation_matrix_weighted.adj')
#     memb,q = bct.community_louvain(A, gamma=1.5)
#     #memb = nq.reindex_membership(memb)
#     #memb = np.loadtxt('Memb1.txt')
#     #draw_parcellation_multiview(template, surf_left, surf_right, memb, 'prova_louvain.png')
#     draw_community_fsaverage()
