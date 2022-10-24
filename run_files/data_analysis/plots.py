import matplotlib.pyplot as plt
from run_files.data_analysis.get_stats import *


def bar_plots_comparing_a_pair_of_means(values, errors, labels, title, save_file=None):
    x_pos = np.arange(len(values))

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, values, yerr=errors, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Average Number of Soups Delivered')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file)
    plt.show()


def bar_plots_comparing_multiple_pairs_of_means(means_1, stds_1, means_2, stds_2, name_1, name_2, labels, title,
                                                save_file=None):
    means_1 = [round(val, 2) for val in means_1]
    means_2 = [round(val, 2) for val in means_2]
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, means_1, width, yerr=stds_1, label=name_1, alpha=0.5, ecolor='black', capsize=2)
    rects2 = ax.bar(x + width/2, means_2, width, yerr=stds_2, label=name_2, alpha=0.5, ecolor='black', capsize=2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Soups Delivered')
    ax.set_title(title)
    ax.yaxis.grid(True, linestyle=':', linewidth=1)
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    if save_file is not None:
        plt.savefig(save_file)
    plt.show()


def compare_multiple_trios_of_means(means_1, stds_1, name_1,
                                    means_2, stds_2, name_2,
                                    means_3, stds_3, name_3,
                                    labels, title, save_file=None,
                                    figsize=None):
    means_1 = [round(val, 2) for val in means_1]
    means_2 = [round(val, 2) for val in means_2]
    means_3 = [round(val, 2) for val in means_3]
    x = np.arange(len(labels))  # the label locations
    width = 0.23  # the width of the bars

    fig, ax = plt.subplots(figsize=figsize)
    rects1 = ax.bar(x - width, means_1, width, yerr=stds_1, label=name_1, alpha=0.5, ecolor='black', capsize=2)
    rects2 = ax.bar(x, means_2, width, yerr=stds_2, label=name_2, alpha=0.5, ecolor='black', capsize=2)
    rects3 = ax.bar(x + width, means_3, width, yerr=stds_3, label=name_3, alpha=0.5, ecolor='black', capsize=2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Soups Delivered')
    ax.set_title(title)
    ax.yaxis.grid(True, linestyle=':', linewidth=1)
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)

    fig.tight_layout()

    if save_file is not None:
        plt.savefig(save_file)
    plt.show()


def plot_beliefs_dpp(save_file=None):
    ydata1 = get_avg_distilled_pp_prob_correct_team()
    ydata2 = get_avg_distilled_pp_prob_correct_team_is_max()

    xdata = np.arange(501)

    # plot the data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xdata, ydata2, color='tab:blue', label="Probability that correct teammate is max")
    ax.plot(xdata, ydata1, color='tab:orange', linestyle='--', label="Probability of correct teammate")
    ax.legend(loc="lower right")

    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Probability')

    ax.set_xlim([0, 500])
    ax.set_ylim([0.17, 1.03])

    ax.set_title('Distilled PLASTIC-Policy Beliefs')

    if save_file is not None:
        plt.savefig(save_file)
    # display the plot
    plt.show()


def plot_beliefs_pp(save_file=None):
    ydata1 = get_avg_pp_prob_correct_team()
    ydata2 = get_avg_pp_prob_correct_team_is_max()

    xdata = np.arange(501)

    # plot the data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xdata, ydata2, color='tab:blue', label="Probability that correct teammate is max")
    ax.plot(xdata, ydata1, color='tab:orange', linestyle='--', label="Probability of correct teammate")
    ax.legend(loc="lower right")

    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Probability')

    ax.set_xlim([0, 500])
    ax.set_ylim([0.17, 1.03])

    ax.set_title('PLASTIC-Policy Beliefs')

    if save_file is not None:
        plt.savefig(save_file)
    # display the plot
    plt.show()


def plot_histogram(title, values):
    n_bins = np.arange(0, values.max() + 1.5) - 0.5
    fig, ax = plt.subplots()
    _ = ax.hist(values, n_bins)
    ax.set_xticks(n_bins + 0.5)

    plt.xlabel('Rewards')
    plt.ylabel('Episodes')
    plt.title(title)
    plt.show()


def plot_histograms(values, teammate_labels, figsize=(20, 3)):
    n_teammates = len(teammate_labels)
    fig, ax = plt.subplots(1, n_teammates, figsize=figsize)

    for i in range(n_teammates):

        n_bins = np.arange(0, values[i].max() + 1.5) - 0.5

        ax[i].hist(values[i].flatten(), n_bins)
        ax[i].set_xlabel('Rewards')
        if i < 1:
            ax[i].set_ylabel('Episodes')
        ax[i].set_title(teammate_labels[i])

    plt.show()