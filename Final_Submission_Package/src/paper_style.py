import matplotlib.pyplot as plt
import matplotlib as mpl
import os

def set_paper_style():
    """Sets matplotlib params for publication-quality figures."""
    # Reset to defaults
    mpl.rcdefaults()
    
    # Fonts
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.titlesize'] = 9 # Panel titles
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7
    plt.rcParams['legend.fontsize'] = 7
    plt.rcParams['legend.title_fontsize'] = 7
    
    # Lines & Markers
    plt.rcParams['lines.linewidth'] = 1.2
    plt.rcParams['lines.markersize'] = 3
    plt.rcParams['lines.markeredgewidth'] = 0.5
    
    # Layout
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.autolayout'] = False # We use tight_layout manually
    plt.rcParams['figure.figsize'] = (3.5, 2.5) # Default single col (~89mm)
    
    # Ticks
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['ytick.major.size'] = 3
    
    # Savefig
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.05
    plt.rcParams['pdf.fonttype'] = 42 # TrueType
    plt.rcParams['ps.fonttype'] = 42

def make_figure(width='single', height_mm=None, nrows=1, ncols=1):
    """
    Creates a figure with specified width (mm-based standard).
    'single': 89mm (3.5 inch)
    'double': 183mm (7.2 inch)
    """
    set_paper_style()
    
    if width == 'single': w_inch = 3.5
    elif width == 'double': w_inch = 7.2
    else: w_inch = width / 25.4
    
    if height_mm is None:
        h_inch = w_inch * 0.75 # Default aspect
    else:
        h_inch = height_mm / 25.4
        
    fig, axes = plt.subplots(nrows, ncols, figsize=(w_inch, h_inch), constrained_layout=True)
    return fig, axes

def save_figure(fig, name, output_dir="outputs/figures/paper"):
    """Saves figure in PDF and PNG formats."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Tight layout first
    # fig.tight_layout(pad=0.5)
    
    # Paths
    pdf_path = os.path.join(output_dir, f"{name}.pdf")
    png_path = os.path.join(output_dir, f"{name}.png")
    
    fig.savefig(pdf_path, format='pdf')
    fig.savefig(png_path, format='png', dpi=600)
    print(f"Saved: {pdf_path} & {png_path}")
    
    # Close to free memory
    plt.close(fig)

def add_panel_label(ax, label, x=-0.1, y=1.05):
    """Adds a bold panel label (A, B, C) to the axes."""
    ax.text(x, y, label, transform=ax.transAxes, 
            fontsize=10, fontweight='bold', va='top', ha='right')
