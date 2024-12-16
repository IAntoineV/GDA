import numpy as np
from matplotlib.patches import Ellipse

def plot_cov_ellipse(cov, mean, ax, color, alpha=0.3):
    """Plot an ellipse to represent the covariance matrix
    Parameters:
    cov (np.ndarray): The covariance matrix
    mean (np.ndarray): The mean of the distribution
    ax (matplotlib.axes.Axes): The axes to plot the ellipse on
    color (str): The color of the ellipse
    alpha (float): The transparency of the ellipse
    
    """
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    u = w[0] / np.linalg.norm(w[0])

    # Angle of the ellipse
    angle = np.arctan2(u[1], u[0])
    angle = np.degrees(angle) 

    # Create the ellipse patch
    ell = Ellipse(xy=mean, width=v[0], height=v[1], angle=angle, color=color, alpha=alpha)

    # Add the ellipse to the plot
    ax.add_patch(ell)
    