import matplotlib.pyplot as plt
import numpy as np

def parallel_plot(array_list, X, plot_kwargs_list=None, title=None, xlabel=None, ylabel=None, label_list=None, save_file=None, ax : plt.Axes = None, n_col=None, n_row = None, param_fig=None, param_legend=None, patche_legend =None, param_fill=None, step=False, marker = True):
    """
    Plots multiple arrays in parallel on the same plot.

    Parameters:
        array_list (list of arrays): A list of arrays to be plotted.
        plot_kwargs_list (list of dict, optional): A list of dictionaries specifying the plot properties for each array.
            If not provided, the default plot properties are used, with different colors for each array.
        title (str, optional): The title for the plot.
        xlabel (str, optional): The label for the x-axis.
        ylabel (str, optional): The label for the y-axis.
        label_list (list of str, optional): A list of labels for each array to be used in the legend.
        save_file (str, optional): The file path to save the plot as a PDF file.
        ax (matplotlib.axes.Axes, optional): The axes object to plot on. If not provided, a new figure and axis is created.

    Returns:
        None
    """

    if ax is None:
        fig, ax = plt.subplots(**param_fig)
        is_ax = False
    else:
        # ax = ax[n_row,n_col]
        is_ax = True

    # If plot_kwargs_list is not provided, use default plot properties with different colors for each array
    if plot_kwargs_list is None:
        plot_kwargs_list = [{'color': f'C{i}'}  for i in range(len(array_list))]

    # Loop through each sublist in array_list and plot the arrays in parallel
    for k, arr in enumerate(array_list):
        # Combine the default plot properties with the plot properties for the current array
        plot_kwargs = {'color': f'C{0}'} | plot_kwargs_list[k]
        # If a label list is provided and the current array has a corresponding label, use it in the plot
        if not step:
            if label_list and k < len(label_list):
                ax.plot(X, arr, label=label_list[k], **plot_kwargs)
            else:
                ax.plot(X, arr,  **plot_kwargs)
            if param_fill:
                max_arr = np.maximum(array_list[0], arr)
                ax.fill_between(X, array_list[0], max_arr, **param_fill)
        else:
            if label_list and k < len(label_list):
                ax.step(X, arr, label=label_list[k], where='post', **plot_kwargs)
            else:
                ax.step(X, arr, where='post', **plot_kwargs)
            if param_fill:
                max_arr = np.maximum(array_list[0], arr)
                ax.fill_between(X, array_list[0], max_arr, **param_fill)

    # Set the plot title, x-axis label, and y-axis label if they are provided
    if title:
        ax.set_title(title)    


    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    


    # Add the legend if a label list is provided
    if not is_ax :
        if label_list or patche_legend:
            handles, labels = ax.get_legend_handles_labels()
            if not marker :
                for h in handles:
                    h.set_marker("")
            if patche_legend :
                handles = patche_legend + handles
                
            ax.legend(handles=handles, **param_legend)
    # Save the plot to a file if save_file is provided
    if save_file:
        fig.savefig(save_file + '.pdf', format='pdf')

    if is_ax:
        return ax

    # Show the plot if save_file is not provided
    else:
        plt.show()

def multi_parallel_plot(array_list, X, plot_kwargs_list=None, title=None, xlabel=None, ylabel=None, label_list=None, 
                        save_file=None, 
                        ax : plt.Axes = None,
                        n_col=None, n_row = None,
                        param_fig : dict =None,
                        param_legend : dict = None,
                        marker = True,
                        legend = True,
                        step = False):
    """
    Plots multiple arrays in parallel on the same plot.

    Parameters:
        array_list (list of arrays): A list of arrays to be plotted.
        plot_kwargs_list (list of dict, optional): A list of dictionaries specifying the plot properties for each array.
            If not provided, the default plot properties are used, with different colors for each array.
        title (str, optional): The title for the plot.
        xlabel (str, optional): The label for the x-axis.
        ylabel (str, optional): The label for the y-axis.
        label_list (list of str, optional): A list of labels for each array to be used in the legend.
        save_file (str, optional): The file path to save the plot as a PDF file.
        ax (matplotlib.axes.Axes, optional): The axes object to plot on. If not provided, a new figure and axis is created.

    Returns:
        None
    """

    if ax is None:
        fig, ax = plt.subplots(**param_fig)
        is_ax = False
    else:
        # ax = ax[n_row,n_col]
        is_ax = True

    # If plot_kwargs_list is not provided, use default plot properties with different colors for each array
    if plot_kwargs_list is None:
        plot_kwargs_list = [{'color': f'C{i}'}  for i in range(len(array_list))]

    # Loop through each sublist in array_list and plot the arrays in parallel
    for k, sublist in enumerate(array_list):
        for i, arr in enumerate(sublist):
            # Combine the default plot properties with the plot properties for the current array
            plot_kwargs = {'color': f'C{i}'} | plot_kwargs_list[k]
            if not step :
                if k == 0:
                    # If a label list is provided and the current array has a corresponding label, use it in the plot
                    if label_list and i < len(label_list):
                        ax.plot(X, arr, label=label_list[i], **plot_kwargs)
                    else:
                        ax.plot(X, arr,  **plot_kwargs)
                else:
                    ax.plot(X, arr,  **plot_kwargs)
            else :
                if k == 0:
                    # If a label list is provided and the current array has a corresponding label, use it in the plot
                    if label_list and i < len(label_list):
                        ax.step(X, arr, label=label_list[i], where='post', **plot_kwargs)
                    else:
                        ax.step(X, arr, where='post', **plot_kwargs)
                else:
                    ax.step(X, arr, where='post', **plot_kwargs)

    # Set the plot title, x-axis label, and y-axis label if they are provided
    if title:
        ax.set_title(title)    


    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Add the legend if a label list is provided
    if not is_ax :
        if label_list or legend:
            ax.legend(**param_legend)

    # Save the plot to a file if save_file is provided
    if save_file:
        fig.savefig(save_file + '.pdf', format='pdf')

    if is_ax:
        return ax

    # Show the plot if save_file is not provided
    else:
        plt.show()
