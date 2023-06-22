#This document holds functions related to creating plots used in my research

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from astropy.visualization import hist as fancyhist


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    covariance = np.cov(x, y)
    pearson = covariance[0, 1]/ np.sqrt(covariance[0, 0] * covariance[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(covariance[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(covariance[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_param_conf(x,y,axes,labels,title,legend_loc='best'):
    '''
    Plots the confidence ellipses for the parameters of recovered lenses

    Parameters
    ----------
    x : array-like, shape (n, )
        The x coordinates of the points.

    y : array-like, shape (n, )
        The y coordinates of the points.

    axes : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    labels : array-like, shape (n, )
        The labels of the points.

    title : string
        The title of the plot

    legend_loc : string
        The location of the legend

    Returns
    -------
    matplotlib.axes.Axes
    '''
    axes.scatter(x,y,marker='.',color='k',label=labels[0])
    confidence_ellipse(x,y,axes,n_std=1,edgecolor='r',label='1$\sigma$')
    confidence_ellipse(x,y,axes,n_std=2,edgecolor='b',label='2$\sigma$',linestyle='--')
    confidence_ellipse(x,y,axes,n_std=3,edgecolor='g',label='3$\sigma$',linestyle=':')
    axes.axvline(0, color = 'black', alpha = 0.5)
    axes.axhline(0, color = 'black', alpha = 0.5)
    axes.set_xlabel(labels[1])
    axes.set_ylabel(labels[2])
    axes.set_title(title)
    axes.legend(loc=legend_loc)
    return axes


def correlation_plot(data,labels,axes):
    N = len(data)
    for i in range(N):
        for j in range(N):
            if i == j:               
                fancyhist(data[i], ax=axes[i,j], bins='freedman', histtype='step', density=True, color='k')
            elif i > j:
                confidence_ellipse(data[i],data[j],axes[i,j],n_std=1,edgecolor='r',label='1$\sigma$')
                confidence_ellipse(data[i],data[j],axes[i,j],n_std=2,edgecolor='b',label='2$\sigma$',linestyle='--')
                confidence_ellipse(data[i],data[j],axes[i,j],n_std=3,edgecolor='g',label='3$\sigma$',linestyle=':')
                axes[i,j].axvline(0, color = 'black', alpha = 0.5)
                axes[i,j].axhline(0, color = 'black', alpha = 0.5)
            else:
                axes[i,j].axis('off')
    
    for i in range(N):
        axes[i,0].set_ylabel(labels[i])
        axes[N-1,i].set_xlabel(labels[i])
    return axes


def plot_likelihood_map(axes,map,lenses,sources,xmax,ymax,zmax,eR,scale,title,legend_loc='best'):
    '''
    Plots the likelihood maps of a lensing field with the locations of the lenses and sources
    along with the locations of the maxima

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    map : array-like, shape (n, )
        The likelihood map.

    lenses : array-like, shape (n, )
        Lenses object, containing the x and y coordinates of the lenses.

    sources : array-like, shape (n, )
        Sources object, containing the x and y coordinates of the sources.

    xmax : array-like, shape (n, )
        x coordinates of the maxima.

    ymax : array-like, shape (n, )
        y coordinates of the maxima.

    zmax : array-like, shape (n, ) 
        strength of the maxima.

    eR : array-like, shape (n, )
        The eR values corresponding to the maxima.

    scale : float
        The number of standard deviations to determine the ellipse's radiuses.

    title : string
        The title of the plot

    legend_loc : string
        The location of the legend

    Returns
    -------
    matplotlib.axes.Axes
    '''

    axes.imshow(map, cmap='RdYlBu_r', extent=[-scale,scale,-scale,scale])

    if sources == None:
        pass
    else:
        axes.scatter(sources.x,sources.y,marker='x',color='k', alpha=0.5, linewidth=2, label='Sources')

    if lenses == None:
        pass
    else:
        axes.scatter(lenses.x,lenses.y,marker='*',color='green', alpha=0.75, linewidth=2, label='Lenses')

    axes.scatter(xmax,ymax,marker='.',color = 'k', s = zmax, linewidth=2, label='Maxima')
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_title(title)
    axes.legend(loc=legend_loc)

    #Beneath each plot, print a table with the maxima's eR values, their 
    #corresponding x and y coordinates, and their strengths
    Nmaxima = len(xmax)
    col_labels = [r"$\theta_E$ ('')", "x('')", "y('')", "P"]
    table_vals = np.zeros((Nmaxima,4)) 
    table_vals[:,0] = eR
    table_vals[:,1] = xmax
    table_vals[:,2] = ymax
    table_vals[:,3] = zmax
    table_vals = np.round(table_vals, decimals=2)

    #Sort the table by the z values
    table_vals = table_vals[table_vals[:,3].argsort()[::-1]]
    #Only keep the top 10 maxima
    table_vals = table_vals[:10]

    data_table = axes.table(cellText=table_vals, 
               colLabels=col_labels, 
               loc='bottom', 
               bbox=[0, -0.75, 1, 0.5])
    
    data_table.auto_set_font_size(False)
    data_table.set_fontsize(10)

    return axes