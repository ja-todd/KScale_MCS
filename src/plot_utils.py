import matplotlib.pyplot as plt 


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