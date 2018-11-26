import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker
plt.switch_backend("Agg")


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def plot_metrics(metrics_list, iterations_list, types, savefile,
                 metric_names=None, n_cols=2, legend=False, x_label=None,
                 y_label=None, wspace=None, hspace=None, figsize=8,
                 bot=None):

    assert isinstance(metrics_list, (list, tuple)) and \
        not isinstance(metrics_list, str)
    # assert bot is None or isinstance(bot, )

    total_n_plots = len(metrics_list)
    if total_n_plots == 1:
        grid_cols, grid_rows = 1, 1
    elif total_n_plots == 2:
        grid_cols, grid_rows = 2, 1
    elif total_n_plots == 3 or total_n_plots == 4:
        grid_cols, grid_rows = 2, 2
    elif total_n_plots == 5 or total_n_plots == 6:
        grid_cols, grid_rows = 2, 3
    elif total_n_plots == 7 or total_n_plots == 8 or total_n_plots == 9:
        grid_cols, grid_rows = 3, 3
    elif total_n_plots == 10 or total_n_plots == 11 or total_n_plots == 12:
        grid_cols, grid_rows = 3, 4
    elif total_n_plots == 13 or total_n_plots == 14 or total_n_plots == 15:
        grid_cols, grid_rows = 3, 5
    elif total_n_plots == 16:
        grid_cols, grid_rows = 4, 4
    elif total_n_plots == 17 or total_n_plots == 18 or \
            total_n_plots == 19 or total_n_plots == 20:
        grid_cols, grid_rows = 4, 5

    fig_w, fig_h = figsize * grid_cols, figsize * grid_rows

    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = GridSpec(grid_rows, grid_cols)
    if wspace is not None and hspace is not None:
        gs.update(wspace=wspace, hspace=hspace)
    elif wspace is not None:
        gs.update(wspace=wspace)
    elif hspace is not None:
        gs.update(hspace=hspace)

    # sorted_alphabetically = np.argsort(metric_names)

    for ii, metric in enumerate(metrics_list):
    # for ii in sorted_alphabetically:
    #     metric = metrics_list[ii]

        current_cell = gs[ii // grid_cols, ii % grid_cols]
        ax = None

        if types[ii] == 'lines':
            ax = plt.subplot(current_cell)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            if isinstance(metric[0], (list, tuple, np.ndarray)):
                lines = []
                for jj, submetric in enumerate(metric):
                    if metric_names is not None:
                        label = metric_names[ii][jj]
                    else:
                        label = "line_%01d" % jj
                    line, = ax.plot(iterations_list[ii], submetric,
                                    color='C%d' % jj,
                                    label=label)
                    lines.append(line)
            else:
                if metric_names is not None:
                    label = metric_names[ii]
                else:
                    label = "line_01"
                line, = ax.plot(iterations_list[ii], metric, color='C0',
                                label=label)
                lines = [line]
            if ((not isinstance(legend, (list, tuple)) and legend) or
                    (isinstance(legend, (list, tuple)) and legend[ii])):
                lg = ax.legend(handles=lines, prop={'size': 16})

        elif types[ii] == 'scatter':
            ax = plt.subplot(current_cell)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            cmap = cm.tab10
            category_labels = metric[..., 2] # [-1]
            norm = colors.Normalize(vmin=np.min(category_labels),
                                    vmax=np.max(category_labels))
            cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
            mapped_colors = cmapper.to_rgba(category_labels)
            unique_labels = list(set(category_labels))
            lines = ax.scatter(metric[..., 0], metric[..., 1], # [-1]
                               color=mapped_colors,
                               label=unique_labels)
            patch = mpatches.Patch(color='silver', label=metric_names[ii])
            ax.legend(handles=[patch], prop={'size': 20})

        elif types[ii] == 'image-grid':
            imgs = metric #[-1]
            n_images = len(imgs)
            inner_grid_width = int(np.sqrt(n_images))
            inner_grid = \
                GridSpecFromSubplotSpec(inner_grid_width,
                                        inner_grid_width, current_cell,
                                        wspace=0.1, hspace=0.1)
            for i in range(n_images):
                inner_ax = plt.subplot(inner_grid[i])
                if imgs.ndim == 4:
                    inner_ax.imshow(imgs[i, :, :, :],
                                    interpolation='none',
                                    vmin=0.0, vmax=1.0)
                else:
                    inner_ax.imshow(imgs[i, :, :], cmap='gray',
                                    interpolation='none',
                                    vmin=0.0, vmax=1.0)
                inner_ax.axis('off')

        if ax is not None:
            if x_label is not None and \
                    not isinstance(x_label, (list, tuple)):
                ax.set_xlabel(x_label, color='k')
            elif isinstance(x_label, (list, tuple)):
                ax.set_xlabel(x_label[ii], color='k')

            # Make the y-axis label, ticks and tick labels
            # match the line color.
            if y_label is not None and \
                    not isinstance(y_label, (list, tuple)):
                ax.set_ylabel(y_label, color='k')
            elif isinstance(y_label, (list, tuple)):
                ax.set_ylabel(y_label[ii], color='k')
            ax.tick_params('y', colors='k')

    plt.savefig(savefile, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if bot is not None:
        bot.send_file(savefile)
