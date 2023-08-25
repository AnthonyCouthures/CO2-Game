import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections


def transpose_plot(fig):
    """
    Transpose a given matplotlib figure along with axis labels.
    
    Parameters:
        fig (matplotlib.figure.Figure): The figure to transpose.
    """
    # Get all axes in the figure
    axes = fig.get_axes()
    
    # Transpose the data on each axis
    for ax in axes:
        lines = ax.lines
        for line in lines:
            xdata, ydata = line.get_xdata(), line.get_ydata()
            line.set_xdata(ydata)
            line.set_ydata(xdata)
            # Transpose scatter plots
        for collection in ax.collections:
            if isinstance(collection, matplotlib.collections.PathCollection):
                offsets = collection.get_offsets()
                collection.set_offsets(np.column_stack((offsets[:, 1], offsets[:, 0])))
        
        # Store the current labels, ticks, limits, scales, and grids
        current_xlabel = ax.get_xlabel()
        current_ylabel = ax.get_ylabel()
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xscale = ax.get_xscale()
        yscale = ax.get_yscale()


        
        # Transpose the axis properties
        ax.set_xlabel(current_ylabel)
        ax.set_ylabel(current_xlabel)
        ax.set_xticks(yticks)
        ax.set_yticks(xticks)
        ax.set_xlim(ylim)
        ax.set_ylim(xlim)
        ax.set_xscale(yscale)
        ax.set_yscale(xscale)




def transpose_subplot(ax):
    """
    Transpose a given matplotlib axis (subplot).
    
    Parameters:
        ax (matplotlib.axes._subplots.AxesSubplot): The subplot to transpose.
    """
    # Transpose the line plots
    for line in ax.lines:
        xdata, ydata = line.get_xdata(), line.get_ydata()
        line.set_xdata(ydata)
        line.set_ydata(xdata)

    # Transpose scatter plots
    for collection in ax.collections:
        if isinstance(collection, matplotlib.collections.PathCollection):
            offsets = collection.get_offsets()
            collection.set_offsets(np.column_stack((offsets[:, 1], offsets[:, 0])))

    # Store and transpose the rest of the axis properties
    current_xlabel = ax.get_xlabel()
    current_ylabel = ax.get_ylabel()
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xscale = ax.get_xscale()
    yscale = ax.get_yscale()


    ax.set_xlabel(current_ylabel)
    ax.set_ylabel(current_xlabel)
    ax.set_xticks(yticks)
    ax.set_yticks(xticks)
    ax.set_xlim(ylim)
    ax.set_ylim(xlim)
    ax.set_xscale(yscale)
    ax.set_yscale(xscale)


    ax.figure.canvas.draw()


def transpose_figure(fig):
    """
    Transpose all subplots within a given matplotlib figure.
    Also swaps supxlabel and supylabel, and updates the suptitle.
    
    Parameters:
        fig (matplotlib.figure.Figure): The figure to transpose.
    """
    # Transpose each subplot in the figure
    for ax in fig.get_axes():
        transpose_subplot(ax)

    # Swap supxlabel and supylabel
    supxlabel_text = ""
    supylabel_text = ""

    if fig._supxlabel:
        supxlabel_text = fig._supxlabel.get_text()
        fig._supxlabel.set_text("")
    if fig._supylabel:
        supylabel_text = fig._supylabel.get_text()
        fig._supylabel.set_text("")

    fig.supxlabel(supylabel_text)
    fig.supylabel(supxlabel_text)

    # You can also swap other super annotations similarly. 
    # For this example, we just dealt with supxlabel and supylabel.
    # If you add other super annotations, you can handle them in the same way.
    
    fig.canvas.draw()