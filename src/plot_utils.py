import matplotlib.pyplot as plt 
import src.hp_models as models
import numpy as np

UNITS = {
    'shear': r'm s$^{-1}$', 
    'tbdiff': r'K', 
    'cr': r'mm hr$^{-1}$',
    'pr': r'mm hr$^{-1}$',
    'condensation_rate': r'kg m$^{-2}$ s$^{-1}$', 
    'surface_precip': r'kg m$^{-2}$ s$^{-1}$', 
}

def apply_plot_style(): 
    """
    Consistently format plots using rcparams
    """


    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 15,
        'axes.labelsize': 16,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.top': False,       # no top ticks (tmXTOn = False)
        'ytick.right': False,
        'ytick.left': True,     # no right ticks (tmYROn = False)
        'axes.linewidth': 1.5,
        'lines.linewidth': 2,
        'axes.spines.top': False, 
        'axes.spines.right': False,
        'axes.facecolor':  "#EAEAF2E6",
        
        
        'axes.grid': True, 
        'grid.color': '#DEDFE4',
        'grid.alpha': 0.5,
        'figure.labelsize': '15', 
        'font.weight': 'normal', 
        'legend.handlelength': 2, 
        'legend.handletextpad': 0.5, 
        'legend.frameon': False,
        'savefig.format': 'pdf', 
        'savefig.dpi': 300, 
        'savefig.bbox': 'tight'
    })

def match_colorbar_to_axes(fig, cbar, axs):
    fig.canvas.draw()  # finalise layout first
    
    all_axes = axs.flatten()
    boxes = [ax.get_position() for ax in all_axes]
    
    y_min = min(box.y0 for box in boxes)
    y_max = max(box.y1 for box in boxes)
    x_max = max(box.x1 for box in boxes)
    
    cbar.ax.set_position([
        x_max + 0.02,   # just to the right of the axes
        y_min,
        0.02,            # colorbar width
        y_max - y_min    # match full height of axes
    ])

def plot_var_setup(region='wam'): 
    """
    Wrapper to make code to set variables for plotting
    at the start of files shorter 
    """
    models_dict = models.models_name_dict
    model_names = list(models_dict.keys())
    region_cfg = models.REGIONS[f'{region}']
    colors = [models.models_name_dict[mname]['color'] for mname in model_names]

    return model_names, region_cfg, colors

def label_subplots(axs, x_offset=-0.1, y_offset=1.02):
    """
    Labels subplots with a), b), c) etc. positioned outside the top left of each axis.
    
    axs: 2D or 1D array of axes (e.g. from plt.subplots)
    x_offset: horizontal position relative to axes (negative = to the left)
    y_offset: vertical position relative to axes (>1 = above)
    """
    axs_flat = np.array(axs).flatten()
    fontsize = plt.rcParams['axes.titlesize']
    
    for i, ax in enumerate(axs_flat):
        label = f'{chr(97 + i)})'  # a), b), c) ...
        ax.text(x_offset, y_offset, label,
                transform=ax.transAxes,
                fontweight='bold',
                fontsize=fontsize,
                ha='left',
                va='bottom')


def centre_legend_above(fig, axs, ncols=4, y_offset=1.02, **legend_kwargs):
    """
    Places a figure legend centred above all subplots.
    
    fig: matplotlib figure
    axs: array of axes
    ncols: number of legend columns
    y_offset: vertical position in figure coordinates (>1 = above axes)
    **legend_kwargs: passed to fig.legend
    """
    fig.canvas.draw()
    
    axs_flat = np.array(axs).flatten()
    
    x_left   = min(ax.get_position().x0 for ax in axs_flat)
    x_right  = max(ax.get_position().x1 for ax in axs_flat)
    y_top    = max(ax.get_position().y1 for ax in axs_flat)
    x_centre = (x_left + x_right) / 2
    
    legend = fig.legend(
        loc='lower center',
        bbox_to_anchor=(x_centre, y_top + (y_offset - 1)),
        ncols=ncols,
        frameon=False,
        **legend_kwargs
    )
    
    return legend