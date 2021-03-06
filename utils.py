import os
import itertools
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, label_binarize

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

    total_n_plots = len(metrics_list)

    # if total_n_plots == 1:
    #     grid_cols, grid_rows = 1, 1
    # elif total_n_plots == 2:
    #     grid_cols, grid_rows = 2, 1
    # elif total_n_plots == 3 or total_n_plots == 4:
    #     grid_cols, grid_rows = 2, 2
    # elif total_n_plots == 5 or total_n_plots == 6:
    #     grid_cols, grid_rows = 2, 3
    # elif total_n_plots == 7 or total_n_plots == 8 or total_n_plots == 9:
    #     grid_cols, grid_rows = 3, 3
    # elif total_n_plots == 10 or total_n_plots == 11 or total_n_plots == 12:
    #     grid_cols, grid_rows = 3, 4
    # elif total_n_plots == 13 or total_n_plots == 14 or total_n_plots == 15:
    #     grid_cols, grid_rows = 3, 5
    # elif total_n_plots == 16:
    #     grid_cols, grid_rows = 4, 4
    # elif total_n_plots == 17 or total_n_plots == 18 or \
    #         total_n_plots == 19 or total_n_plots == 20:
    #     grid_cols, grid_rows = 4, 5

    grid_cols = int(np.ceil(np.sqrt(total_n_plots)))
    grid_rows = int(np.ceil(total_n_plots / grid_cols))

    fig_w, fig_h = figsize * grid_cols, figsize * grid_rows

    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = GridSpec(grid_rows, grid_cols)
    if wspace is not None and hspace is not None:
        gs.update(wspace=wspace, hspace=hspace)
    elif wspace is not None:
        gs.update(wspace=wspace)
    elif hspace is not None:
        gs.update(hspace=hspace)

    for ii, metric in enumerate(metrics_list):
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
            category_labels = metric[..., 2]
            norm = colors.Normalize(vmin=np.min(category_labels),
                                    vmax=np.max(category_labels))
            cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
            mapped_colors = cmapper.to_rgba(category_labels)
            unique_labels = list(set(category_labels))
            lines = ax.scatter(metric[..., 0], metric[..., 1],
                               color=mapped_colors,
                               label=unique_labels)
            patch = mpatches.Patch(color='silver', label=metric_names[ii])
            ax.legend(handles=[patch], prop={'size': 20})

        elif types[ii] == 'image-grid':
            imgs = metric
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
        elif types[ii] == 'confusion-matrix':
            ax = plt.subplot(current_cell)
            cmap = plt.cm.Blues
            cm_norm = metric[0].astype('float') \
                / metric[0].sum(axis=1)[:, np.newaxis]
            im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap)
            # cbar = ax.colorbar()
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Percentage', rotation=90)
            tick_marks = np.arange(len(metric[1]))
            plt.sca(ax)
            plt.title(metric_names[ii])
            plt.xticks(tick_marks, metric[1], rotation=45)
            plt.yticks(tick_marks, metric[1])

            fmt = 'd'
            thresh = cm_norm.max() / 2.
            for i, j in itertools.product(range(metric[0].shape[0]),
                                          range(metric[0].shape[1])):
                ax.text(j, i, format(metric[0][i, j], fmt),
                        horizontalalignment='center',
                        color="white" if cm_norm[i, j] > thresh else 'black')

            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()

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


def plot_metrics_individually(metrics_list, iterations_list, types, savefile,
                              metric_names=None, n_cols=2, legend=False,
                              x_label=None, y_label=None, wspace=None,
                              hspace=None, figsize=8, bot=None):

    assert isinstance(metrics_list, (list, tuple)) and \
        not isinstance(metrics_list, str)

    total_n_plots = len(metrics_list)

    for ii, metric in enumerate(metrics_list):
        fig = plt.figure(figsize=(figsize, figsize))

        if types[ii] == 'lines':
            ax = fig.add_subplot(1, 1, 1)
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
            ax = fig.add_subplot(1, 1, 1)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            cmap = cm.tab10
            category_labels = metric[..., 2]
            norm = colors.Normalize(vmin=np.min(category_labels),
                                    vmax=np.max(category_labels))
            cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
            mapped_colors = cmapper.to_rgba(category_labels)
            unique_labels = list(set(category_labels))
            lines = ax.scatter(metric[..., 0], metric[..., 1],
                               color=mapped_colors,
                               label=unique_labels)
            patch = mpatches.Patch(color='silver', label=metric_names[ii])
            ax.legend(handles=[patch], prop={'size': 20})

        elif types[ii] == 'image-grid':
            imgs = metric
            n_images = len(imgs)
            inner_grid_width = int(np.sqrt(n_images))
            gs = GridSpec(inner_grid_width, inner_grid_width,
                          wspace=0.1, hspace=0.1)
            # inner_grid = \
            #     GridSpecFromSubplotSpec(inner_grid_width,
            #                             inner_grid_width, current_cell,
            #                             wspace=0.1, hspace=0.1)

            for jj in range(n_images):
                current_cell = gs[jj // inner_grid_width,
                                  jj % inner_grid_width]
                ax = plt.subplot(current_cell)
                if imgs.ndim == 4:
                    ax.imshow(imgs[jj, :, :, :],
                              interpolation='none',
                              vmin=0.0, vmax=1.0)
                else:
                    ax.imshow(imgs[jj, :, :], cmap='gray',
                              interpolation='none',
                              vmin=0.0, vmax=1.0)
                ax.axis('off')
        elif types[ii] == 'confusion-matrix':
            ax = fig.add_subplot(1, 1, 1)
            cmap = plt.cm.Blues
            cm_norm = metric[0].astype('float') \
                / metric[0].sum(axis=1)[:, np.newaxis]
            im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap)
            # cbar = ax.colorbar()
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Percentage', rotation=90)
            tick_marks = np.arange(len(metric[1]))
            plt.sca(ax)
            plt.title(metric_names[ii])
            plt.xticks(tick_marks, metric[1], rotation=45)
            plt.yticks(tick_marks, metric[1])

            fmt = 'd'
            thresh = cm_norm.max() / 2.
            for i, j in itertools.product(range(metric[0].shape[0]),
                                          range(metric[0].shape[1])):
                ax.text(j, i, format(metric[0][i, j], fmt),
                        horizontalalignment='center',
                        color="white" if cm_norm[i, j] > thresh else 'black')

            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()

        if ax is not None:
            if x_label is not None and not isinstance(x_label, (list, tuple)):
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

        savefile_ind = savefile.rsplit('/', 1)[0] + '/' + metric_names[ii] + \
            '_' + savefile.rsplit('/', 1)[1]

        plt.savefig(savefile_ind,
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        if bot is not None:
            bot.send_file(savefile_ind)


def get_labels(one_hot_labels, n_labels=2):
    labels = [np.nonzero(one_hot_labels[ii, :])[0]
              for ii in range(one_hot_labels.shape[0])]

    labels = [np.repeat(labels[ii], n_labels).reshape(1, -1)
              if labels[ii].shape[0] < n_labels
              else labels[ii].reshape(1, -1)
              for ii in range(len(labels))]

    return np.vstack(labels)


class SVMEval():
    """
    Based on:
    https://github.com/leosampaio/keras-generative/blob/master/metrics/svm.py
    """
    def __init__(self, tr_size, val_size, n_splits=5, n_labels=1,
                 scale=False, per_class=True, verbose=0):
        self.tr_size = tr_size
        self.val_size = val_size
        self.n_splits = n_splits
        self.n_labels = n_labels
        self.param_grid = [{'C': [0.1, 1, 10, 100, 1000]}]
        # self.param_grid = [{'alpha': [1 * tr_size,
        #                               0.1 * tr_size,
        #                               0.01 * tr_size,
        #                               0.001 * tr_size]}]
        # C_svc * n_samples = 1 / alpha_sgd
        self.scale = scale
        self.per_class = per_class
        self.verbose = verbose

        if self.n_splits > 1:
            self.acc_grid = GridSearchCV(
                LinearSVC(max_iter=1000, tol=1e-3),
                param_grid=self.param_grid,
                cv=self.n_splits, verbose=self.verbose,
                scoring='balanced_accuracy', n_jobs=12)
            # self.auc_grid = GridSearchCV(
            #     SGDClassifier(max_iter=1000, tol=1e-3),
            #     param_grid=self.param_grid,
            #     cv=self.n_splits, verbose=self.verbose,
            #     scoring='roc_auc', n_jobs=3)
        else:
            raise NotImplementedError

    def compute(self, train_data, val_data):
        x_train, y_train = train_data
        x_test, y_test = val_data

        valid_classes = np.unique(get_labels(y_train, n_labels=self.n_labels))

        if self.scale is True:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

        if self.per_class is True:
            y_train = label_binarize(y_train, classes=list(valid_classes))
            y_test = label_binarize(y_test, classes=list(valid_classes))

            val_acc = []
            val_auc = []
            for cl in range(len(valid_classes)):
                aux_acc = []
                # aux_auc = []
                # if self.n_splits > 1:
                #     # grid = GridSearchCV(
                #     #     LinearSVC(multi_class='ovr'),
                #     #     param_grid=self.param_grid,
                #     #     cv=self.n_splits, verbose=self.verbose,
                #     #     scoring='accuracy', n_jobs=3)
                #     grid = GridSearchCV(
                #         SGDClassifier(),
                #         param_grid=self.param_grid,
                #         cv=self.n_splits, verbose=self.verbose,
                #         scoring='accuracy', n_jobs=3)
                #     grid.fit(X=x_train, y=y_train[:, cl])
                #     acc_on_val = grid.score(x_test, y_test[:, cl])
                # else:
                #     raise NotImplementedError
                self.acc_grid.fit(X=x_train, y=y_train[:, cl])
                pred_on_val_cl_acc = self.acc_grid.predict(x_test)

                if cl == 0:
                    pred_on_val_acc = np.array(pred_on_val_cl_acc)[np.newaxis].T
                else:
                    pred_on_val_acc = np.append(pred_on_val_acc, np.array(pred_on_val_cl_acc)[np.newaxis].T, axis=1)

                acc_on_val = self.acc_grid.score(x_test, y_test[:, cl])
                print(pred_on_val_acc.shape)
                val_acc.append(acc_on_val)
                print(val_acc)

                # if self.n_splits > 1:
                #     # grid = GridSearchCV(LinearSVC(multi_class='ovr'),
                #     #                     param_grid=self.param_grid,
                #     #                     cv=self.n_splits, verbose=self.verbose,
                #     #                     scoring='roc_auc', n_jobs=3)
                #     grid = GridSearchCV(SGDClassifier(),
                #                         param_grid=self.param_grid,
                #                         cv=self.n_splits, verbose=self.verbose,
                #                         scoring='roc_auc', n_jobs=3)
                #     grid.fit(X=x_train, y=y_train[:, cl])
                #     auc_on_val = grid.score(x_test, y_test[:, cl])
                # else:
                #     raise NotImplementedError
                # self.auc_grid.fit(X=x_train, y=y_train[:, cl])
                # pred_on_val_cl_auc = self.auc_grid.predict(x_test)
                #
                # if cl == 0:
                #     pred_on_val_auc = np.array(pred_on_val_cl_auc)[np.newaxis].T
                # else:
                #     pred_on_val_auc = np.append(pred_on_val_auc, np.array(pred_on_val_cl_auc)[np.newaxis].T, axis=1)
                # auc_on_val = self.auc_grid.score(x_test, y_test[:, cl])
                # aux_auc.append(auc_on_val)

                # val_acc.append(np.mean(aux_acc))
                # val_auc.append(np.mean(aux_auc))

            return val_acc, pred_on_val_acc, valid_classes  # val_auc,

        else:
            val_acc = []
            if self.n_splits > 1:
                grid = GridSearchCV(LinearSVC(multi_class='ovr'),
                                    param_grid=self.param_grid,
                                    cv=self.n_splits, verbose=self.verbose,
                                    scoring='accuracy', n_jobs=3)
                grid.fit(X=x_train, y=y_train)
                score_on_val = grid.score(x_test, y_test)
            else:
                raise NotImplementedError
            val_acc.append(score_on_val)

            return np.mean(val_acc)
