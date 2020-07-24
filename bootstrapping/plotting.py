import pickle
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def from_file_to_dict(interim_distrib_pickle):
    with open(interim_distrib_pickle, 'rb') as handle:
        return pickle.load(handle)


def from_dict_to_sample(interim_distrib, upper_bound=np.inf):

    df = pd.DataFrame(data=interim_distrib)
    z = df.mean()

    u = []
    for item in z.iteritems():
        if item[0] < upper_bound:
            u.extend([item[0]] * int(np.round(item[1])))
    return u


def from_sample_to_ecdf(sample):
    return ECDF(sample)


def from_file_to_ecdf(interim_distrib_pickle, upper_bound=np.inf):
    return from_sample_to_ecdf(from_dict_to_sample(from_file_to_dict(interim_distrib_pickle), upper_bound))


def figure1(interim_general, interim_simple, interim):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(11, 8), dpi=300)
    sns.distplot(interim_simple[0], ax=ax[0][0],
                 kde_kws={"color": "b", "lw": 2, 'alpha': 0.5},
                 label='Basic estimation of household size')
    sns.distplot(interim_simple[1], ax=ax[0][1],
                 kde_kws={"color": "b", "lw": 2, 'alpha': 0.5})
    sns.distplot(interim_simple[2], ax=ax[1][0],
                 kde_kws={"color": "b", "lw": 2, 'alpha': 0.5})
    sns.distplot(interim_simple[3], ax=ax[1][1],
                 kde_kws={"color": "b", "lw": 2, 'alpha': 0.5})
    sns.distplot(interim[0], ax=ax[0][0],
                 kde_kws={"color": "r", "lw": 2, 'alpha': 0.5},
                 label='Estimation of household size with partial information on household size')
    sns.distplot(interim[1], ax=ax[0][1],
                 kde_kws={"color": "r", "lw": 2, 'alpha': 0.5})
    sns.distplot(interim[2], ax=ax[1][0],
                 kde_kws={"color": "r", "lw": 2, 'alpha': 0.5})
    sns.distplot(interim[3], ax=ax[1][1],
                 kde_kws={"color": "r", "lw": 2, 'alpha': 0.5})
    sns.distplot(interim_general[0], ax=ax[0][0],
                 kde_kws={"color": "g", "lw": 2, 'alpha': 0.5},
                 label='Estimation of household size including spatial data')
    sns.distplot(interim_general[1], ax=ax[0][1],
                 kde_kws={"color": "g", "lw": 2, 'alpha': 0.5})
    sns.distplot(interim_general[2], ax=ax[1][0],
                 kde_kws={"color": "g", "lw": 2, 'alpha': 0.5})
    sns.distplot(interim_general[3], ax=ax[1][1],
                 kde_kws={"color": "g", "lw": 2, 'alpha': 0.5})
    ax[0][0].set_title('0-39')
    ax[0][1].set_title('40-59')
    ax[1][0].set_title('60-79')
    ax[1][1].set_title('80+')
    handles, labels = ax[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center')
    # plt.suptitle('Distribution of susceptibles in each age group based on the results of bootstrapping procedure')
    plt.subplots_adjust(top=0.95, bottom=0.15, hspace=0.25)
    plt.savefig('susceptibles_distplot.png')
    plt.show()


def plotting2():
    with open('results10000_simplest_20200620_406080_2.pickle', 'rb') as handle:
        interim_simple = pickle.load(handle)

    with open('results10000_households_20200620_406080_2.pickle', 'rb') as handle:
        interim = pickle.load(handle)

    with open('results10000_households_general_20200620_406080_2.pickle', 'rb') as handle:
        interim_general = pickle.load(handle)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(11, 8))
    sns.distplot(interim_simple[0], ax=ax[0][0],
                 kde_kws={"color": "b", "lw": 2, 'alpha': 0.5},
                 label='Basic estimation of household size',
                 rug=True, hist_kws={"histtype": "step", "linewidth": 3, 'alpha': 0.6})
    sns.distplot(interim_simple[1], ax=ax[0][1],
                 kde_kws={"color": "b", "lw": 2, 'alpha': 0.5},
                 rug=True, hist_kws={"histtype": "step", "linewidth": 3, 'alpha': 0.6})
    sns.distplot(interim_simple[2], ax=ax[1][0],
                 kde_kws={"color": "b", "lw": 2, 'alpha': 0.5},
                 rug=True, hist_kws={"histtype": "step", "linewidth": 3, 'alpha': 0.6})
    sns.distplot(interim_simple[3], ax=ax[1][1],
                 kde_kws={"color": "b", "lw": 2, 'alpha': 0.5},
                 rug=True, hist_kws={"histtype": "step", "linewidth": 3, 'alpha': 0.6})
    sns.distplot(interim[0], ax=ax[0][0],
                 kde_kws={"color": "r", "lw": 2, 'alpha': 0.5},
                 label='Estimation of household size including partial information on household size',
                 rug=True, hist_kws={"histtype": "step", "linewidth": 3, 'alpha': 0.6})
    sns.distplot(interim[1], ax=ax[0][1],
                 kde_kws={"color": "r", "lw": 2, 'alpha': 0.5},
                 rug=True, hist_kws={"histtype": "step", "linewidth": 3, 'alpha': 0.6})
    sns.distplot(interim[2], ax=ax[1][0],
                 kde_kws={"color": "r", "lw": 2, 'alpha': 0.5},
                 rug=True, hist_kws={"histtype": "step", "linewidth": 3, 'alpha': 0.6})
    sns.distplot(interim[3], ax=ax[1][1],
                 kde_kws={"color": "r", "lw": 2, 'alpha': 0.5},
                 rug=True, hist_kws={"histtype": "step", "linewidth": 3, 'alpha': 0.6})
    sns.distplot(interim_general[0], ax=ax[0][0],
                 kde_kws={"color": "g", "lw": 2, 'alpha': 0.5},
                 label='Estimation of household size including partial information on household size and spatial data',
                 rug=True, hist_kws={"histtype": "step", "linewidth": 3, 'alpha': 0.6})
    sns.distplot(interim_general[1], ax=ax[0][1],
                 kde_kws={"color": "g", "lw": 2, 'alpha': 0.5},
                 rug=True, hist_kws={"histtype": "step", "linewidth": 3, 'alpha': 0.6})
    sns.distplot(interim_general[2], ax=ax[1][0],
                 kde_kws={"color": "g", "lw": 2, 'alpha': 0.5},
                 rug=True, hist_kws={"histtype": "step", "linewidth": 3, 'alpha': 0.6})
    sns.distplot(interim_general[3], ax=ax[1][1],
                 kde_kws={"color": "g", "lw": 2, 'alpha': 0.5},
                 rug=True, hist_kws={"histtype": "step", "linewidth": 3, 'alpha': 0.6})
    ax[0][0].set_title('0-39')
    ax[0][1].set_title('40-59')
    ax[1][0].set_title('60-79')
    ax[1][1].set_title('80+')
    handles, labels = ax[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center')
    # plt.suptitle('Distribution of susceptibles in each age group based on the results of bootstrapping procedure')
    plt.subplots_adjust(top=1, bottom=0.15)
    plt.savefig('susceptibles_distplot.png')
    plt.show()


def plot_g(G, result_image_file):
    plt.figure(dpi=300)
    plt.plot(G['lambda'], G['G'], label='$G(\lambda)$')
    plt.xlabel('$\lambda$')
    plt.ylabel('Fraction of infected')
    plt.legend()
    plt.savefig(result_image_file)
    plt.show()