import matplotlib.pyplot as plt
import numpy as np
import squidpy as sq

def show_training_curve(epochs, losses=[], loss_labels=[]):
    fig, ax = plt.subplots()
    ax.clear()
    for i in range(len(losses)):
        ax.plot(epochs, losses[i], label=loss_labels[i])
    ax.legend()
    plt.show()

def display_cell_deconvolution(adata, cell_types, ncols=4):
    num_cell_types = len(cell_types)
    nrows = num_cell_types // ncols + (0 if num_cell_types % ncols == 0 else 1)

    values = adata.obs[cell_types].to_numpy()
    vmin = np.min(values)
    vmax = np.max(values)
   
    fig, axs = plt.subplots(nrows, ncols)
    for cell_type, ax in zip(cell_types, axs.flat):
        sq.pl.spatial_scatter(adata, 
                              color=cell_type,
                              fig=fig,
                              ax=ax,
                              vmin=vmin, 
                              vmax=vmax, 
                              colorbar=False,
                              title=cell_type,
                              figsize=6,
                              axis_label="")


    # print(axs)
    # print(axs.ravel())
    print(axs.ravel().tolist())
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, ax=axs.ravel().tolist())
    cbar.set_label('Cell Type Proportion')
    fig.suptitle=('Cell Deconvolution')
    fig.set_size_inches(10,8)
    # print(fig.axes)
    # plt.tight_layout()
    plt.show()

def display_cell_proportions(adata, cell_types, ncols=4):
    num_cell_types = len(cell_types)
    nrows = num_cell_types // ncols + (0 if num_cell_types % ncols == 0 else 1)

    columns = [f"props_{cell_type}" for cell_type in cell_types]
    values = adata.obs[cell_types].to_numpy()
    vmin = np.min(values)
    vmax = np.max(values)
   
    fig, axs = plt.subplots(nrows, ncols)
    for cell_type, ax in zip(cell_types, axs.flat):
        sq.pl.spatial_scatter(adata, 
                              color=cell_type,
                              fig=fig,
                              ax=ax,
                              vmin=vmin, 
                              vmax=vmax, 
                              colorbar=False,
                              title=cell_type,
                              figsize=6,
                              axis_label="")


    # print(axs)
    # print(axs.ravel())
    print(axs.ravel().tolist())
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, ax=axs.ravel().tolist())
    cbar.set_label('Cell Type Proportion')
    fig.suptitle=('Cell Deconvolution')
    fig.set_size_inches(10,8)
    # print(fig.axes)
    # plt.tight_layout()
    plt.show()


    
def display_decomposition(adata_decomposition, gene, cell_types, vmin, vmax, ncols=5):
    num_cell_types = len(cell_types)
    nrows = num_cell_types // ncols + (1 if num_cell_types % ncols == 0 else 0)

    fig, axes = plt.subplots(ncols=ncols)
    for cell_type, ax in zip(cell_types, axes.flat):
        sq.pl.spatial_scatter(adata_decomposition[cell_type], 
                              color=gene, title=f"{gene} ({cell_type})", 
                              vmin=vmin, 
                              vmax=vmax,
                              ax=ax)

    plt.show()
    