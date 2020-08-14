from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from bisect import bisect_right
import random
import pickle
from typing import Any
from bootstrapping.nsp2011 import get_households_within_habitants_range, get_data, get_data_for_voy, \
    get_people_in_households
from bootstrapping.infected_dataset import (get_elderly_patient_data, get_patient_data, get_index_cases_grouped_by_age,
                                            get_known_secondary_infected_count,
                                            get_known_secondary_infected_age_grouped)
from bootstrapping import utils
from bootstrapping.settings import *
from bootstrapping.plotting import plot_g
from operator import itemgetter


class AttackRate:

    def __init__(self, max_household_size=DEFAULT_MAX_MIN_HOUSEHOLD_SIZE) -> None:
        self.dt = datetime.now().strftime('%Y%m%d%H%M')
        self.pickle_file_pattern = f'{{}}_{self.dt}.pickle'
        self.max_household_size = max_household_size
        self.K = self.get_K()
        # self.lambdas = self.get_lambdas()

    def dump_pickles(self, data: Any, file_part: str) -> None:
        with (RESULTS_DIR / self.pickle_file_pattern.format(file_part)).open('wb') as handle:
            pickle.dump(data, handle)

    @staticmethod
    def load_pickles(file_path: Path):
        with file_path.open('rb') as handle:
            return pickle.load(handle)

    @staticmethod
    def get_K(max_household_size=DEFAULT_MAX_MIN_HOUSEHOLD_SIZE):
        return np.array(list(range(max_household_size - 1)))

    # @staticmethod
    # def get_lambdas():
    #     return np.logspace(-10, -2, base=2, num=50)  # more dense at the beginning 0 to 0.15

    @staticmethod
    def get_mu_k_lambda(full_range=False):
        if full_range:
            mu_k_lambda = pd.read_csv(str(data_dir / 'mu_of_alpha_full_range.csv'), index_col=[0])
        else:
            mu_k_lambda = pd.read_csv(str(data_dir / 'mu_of_alpha.csv'), index_col=[0])
        mu_k_lambda.columns.name = 'lambda'
        return mu_k_lambda

    def calculate_probabilities_of_household_size(self, data, elderly_data, household_size_min=1,
                                                  household_size_max=DEFAULT_MAX_MIN_HOUSEHOLD_SIZE,
                                                  num_trials=DEFAULT_NUM_TRIALS,
                                                  cutoff_age=DEFAULT_CUTOFF_AGE):
        age_min = data.age.min()
        age_max = data.age.max() + 1
        age_count = age_max - age_min
        gender_count = len(GENDER_INDICES)
        voy_count = len(voy_mapping)
        household_size_count = household_size_max - household_size_min

        occurrences = np.zeros((voy_count, age_count, gender_count, household_size_count,
                                household_size_count))
        num_cases = np.zeros((voy_count, age_count, gender_count, household_size_count))
        all_people, all_households = get_data()

        for n in tqdm(elderly_data.min_household_size.unique(), desc='Calculating elderly'):

            households = get_households_within_habitants_range(all_households, n, household_size_max)
            people = get_people_in_households(all_people, households)

            for j, row in elderly_data[elderly_data.min_household_size == n].iterrows():
                age = row.age
                gender = row.gender
                for voy in row.voy:
                    num_cases[VOYVODSHIPS.index(voy), age - age_min, gender, n - household_size_min] += 1
                household_indices = random.choices(people[(gender, age)], k=num_trials)
                for index in household_indices:
                    occurrences[:,
                    age - age_min,
                    gender,
                    n - household_size_min,
                    len(households[index]) - household_size_min] += 1

        for voy_idx, voy in enumerate(tqdm(VOYVODSHIPS, desc='Calculating young fellows')):
            all_people, all_households = get_data_for_voy(voy)

            voy_data = data[(data.voy == voy) & (data.age < cutoff_age)]

            for n in voy_data.min_household_size.unique():
                households = get_households_within_habitants_range(all_households, n, household_size_max)
                people = get_people_in_households(all_people, households)

                for gender in GENDER_INDICES:
                    sources_dict = voy_data.loc[(voy_data.gender == gender) & (voy_data.min_household_size == n),
                                                ['age', 'number_of_cases']].set_index('age')[
                        'number_of_cases'].to_dict()
                    for age, number_of_cases in sources_dict.items():
                        num_cases[voy_idx, age - age_min, gender, n - household_size_min] = number_of_cases
                        household_indices = random.choices(people[(gender, age)], k=num_trials)
                        for index in household_indices:
                            occurrences[voy_idx,
                                        age - age_min,
                                        gender,
                                        n - household_size_min,
                                        len(households[index]) - household_size_min] += 1

        occurrences = occurrences / num_trials
        self.dump_pickles(occurrences, 'accommodation_probability')
        self.dump_pickles(num_cases, 'num_cases')
        return occurrences, num_cases

    def get_mu_bar(self, probabilities, mu_k_lambda, lambdas):
        # mu_k_lambda - dataframe, lambdas x household size (K-1)
        # probabilities - matrix (voy, age, gender, min household size, household size)
        lambdas_count = len(lambdas)
        voy_count = probabilities.shape[0]
        age_count = probabilities.shape[1]
        gender_count = probabilities.shape[2]
        household_size_count = probabilities.shape[3]

        mu_bar = np.zeros((lambdas_count, voy_count, age_count, gender_count, household_size_count))

        for lambda_idx, _lambda in enumerate(lambdas):
            for voy_idx in range(voy_count):
                for age_idx in range(age_count):
                    for gender_idx in range(gender_count):
                        for mhs_idx in range(household_size_count):
                            mu_bar[lambda_idx, voy_idx, age_idx, gender_idx, mhs_idx] = np.sum(
                                mu_k_lambda[mu_k_lambda.columns[lambda_idx]].to_numpy() \
                                * probabilities[voy_idx, age_idx, gender_idx, mhs_idx, 1:])

        self.dump_pickles(mu_bar, 'mu_bar')
        return mu_bar

    def get_probabilities_of_infection(self, _lambda):
        def compute_number_of_infected(alpha, household_size, random_number_generator, x0=1):
            infected_matrix = np.zeros(household_size).astype(bool)
            sampled = np.zeros(household_size).astype(bool)
            infected_matrix[:x0] = 1
            while np.any(infected_matrix & ~sampled):
                arr = infected_matrix & ~sampled
                infecting_id = arr.nonzero()[0][0]
                sampled[infecting_id] = True
                probs = np.ones(household_size)
                probs[infecting_id] = 0
                probs /= (household_size - 1)
                infections = random_number_generator.binomial(household_size - 1, alpha)
                choices = random_number_generator.choice(household_size, infections, replace=False, p=probs)
                infected_matrix[choices] = True
            return sum(infected_matrix) - 1

        iterations = 20000
        expected = []
        k_values = np.arange(2, 15)  # this is the size of household
        inf_table = np.zeros((len(k_values), len(k_values) + 1))
        for k in k_values:
            expected_infected = 0
            for iters in np.arange(iterations):
                rng = np.random.default_rng()
                infected = compute_number_of_infected(_lambda, k, random_number_generator=rng)
                inf_table[k - 2, infected] += 1
                expected_infected = (expected_infected * iters + infected) / (iters + 1)

            expected.append(expected_infected)

        inf_table /= iterations
        self.dump_pickles(inf_table, 'probability_of_infection')
        self.dump_pickles(expected, 'expected_infected')

        return inf_table, expected

    @staticmethod
    def get_lambda_hat(g_df, infected, susceptibles):
        arg = infected / susceptibles
        where = bisect_right(g_df.G, arg)
        a = (arg - g_df.loc[where - 1].G) / (g_df.loc[where].G - g_df.loc[where - 1].G)
        lambda_hat = g_df.loc[where - 1, 'lambda'] + a * (g_df.loc[where, 'lambda'] - g_df.loc[where - 1, 'lambda'])
        return lambda_hat

    @staticmethod
    def get_g_function(ei_df, EN_asterisk):
        return (ei_df.set_index('lambda').EI / EN_asterisk).reset_index().rename(columns={'EI': 'G'})

    @staticmethod
    def get_EN_asterisk(bar_h, num_cases):
        return np.sum(np.multiply(bar_h, num_cases))

    def get_bar_h(self, p):
        bar_h = np.sum(p * self.K, axis=-1)
        self.dump_pickles(bar_h, 'bar_h')
        return bar_h

    def get_ei(self, mu_bar, num_cases, lambdas):
        EI = dict()
        for idx, _lambda in enumerate(lambdas):
            EI[_lambda] = np.sum(np.multiply(mu_bar[idx, :, :, :, :], num_cases))
        ei_df = pd.Series(data=EI).reset_index().rename(columns={'index': 'lambda', 0: 'EI'})
        self.dump_pickles(ei_df, 'ei_df')
        return ei_df

    def infect_susceptibles(self, sampled_households, lambda_hat):
        # based on households drawn for 5D case, process infection in the uniform manner (not age dependent)
        prob_table, expected_infected = self.get_probabilities_of_infection(lambda_hat)
        sampled_households = sampled_households.astype(int)
        K = np.arange(0, 14)  # self.K
        num_trials = sampled_households.shape[0]
        num_age_groups = sampled_households.shape[2]
        total_infected = np.zeros((num_trials, num_age_groups), dtype=int)
        for i in tqdm(range(num_trials)):

            for j, household in enumerate(sampled_households[i]):
                # household has 5 columns with number of susceptibles in each age group
                num_susceptibles = sum(household)
                if num_susceptibles == 0:
                    continue
                # draw at random number of secondary infected in this household
                k = np.random.choice(K, p=prob_table[num_susceptibles - 1, :])
                if k == 0:
                    continue
                # construct a vector of susceptibles by their age groups
                susceptibles_age_groups = []
                for age_group, headcount in enumerate(household):
                    susceptibles_age_groups.extend([age_group] * headcount)
                # select at random k infected among all susceptibles
                new_infected = np.random.choice(susceptibles_age_groups, size=k, replace=False)
                # increase the counter for this iteration
                for new_one in new_infected:
                    total_infected[i, new_one] += 1
        # in the process of bisection this will overwrite the results of the preceding iteration, but we actually
        # care only about the last iteration
        self.dump_pickles(total_infected, 'sampled_households_secondary')
        return total_infected

    def sample_households_and_infect(self, data, elderly_data, prob_table, household_size_max=15, num_trials=10000,
                                     cutoff_age=90):
        all_people, all_households = get_data()

        max_age_in_census = max(all_people.keys(), key=itemgetter(1))[1]
        output = np.zeros((num_trials, max_age_in_census + 1))

        K = np.arange(0, 14)  # self.K
        wielkosc_domkow = np.zeros((num_trials, len(K)))

        check_index_cases_count = 0
        for n in tqdm(elderly_data.min_household_size.unique()):

            households = get_households_within_habitants_range(all_households, n, household_size_max)
            people = get_people_in_households(all_people, households)

            for j, row in elderly_data[elderly_data.min_household_size == n].iterrows():
                age = row.age
                gender = row.gender
                voyvodships = row.voy
                number_of_cases = len(voyvodships)
                check_index_cases_count += number_of_cases

                for i in range(num_trials):
                    household_indices = random.choices(people[(gender, age)], k=number_of_cases)
                    assert number_of_cases == len(household_indices)

                    for household_index in household_indices:
                        h = households[household_index]
                        wielkosc_domkow[i, len(h) - 1] += 1
                        if len(h) == 1:
                            continue
                        # if len(h) == 2 then sample from K with probability prob_table[0,:]
                        k = np.random.choice(K, p=prob_table[len(h) - 2, :])

                        pos = h.index((gender, age))
                        susceptibles_ages = [x[1] for idx, x in enumerate(h) if idx != pos]
                        new_infected = np.random.choice(susceptibles_ages, size=k, replace=False)

                        for new_one in new_infected:
                            output[i, new_one] += 1
        print(f'Check: Index cases count: {check_index_cases_count}')
        # print(f'Index cases from db: {len(elderly_data.index)}')

        check_index_cases_count_below90 = 0
        for voy_idx, voy in enumerate(tqdm(voy_mapping.values())):
            all_people, all_households = get_data_for_voy(voy)

            voy_data = data[(data.voy == voy) & (data.age < cutoff_age)]
            # check_index_cases_count_below90 += len(voy_data.index)

            for n in voy_data.min_household_size.unique():
                # print(f'n={n}')
                households = get_households_within_habitants_range(all_households, n, household_size_max)
                people = get_people_in_households(all_people, households)

                for gender in GENDER_INDICES:
                    sources_dict = voy_data.loc[(voy_data.gender == gender) & (voy_data.min_household_size == n),
                                                ['age', 'number_of_cases']] \
                        .set_index('age')['number_of_cases'].to_dict()
                    # print(voy_data.number_of_cases.sum())
                    check_index_cases_count_below90 += sum(sources_dict.values())

                    for age, number_of_cases in sources_dict.items():
                        for i in range(num_trials):

                            household_indices = random.choices(people[(gender, age)], k=number_of_cases)
                            for household_index in household_indices:
                                h = households[household_index]
                                wielkosc_domkow[i, len(h) - 1] += 1
                                if len(h) == 1:
                                    continue
                                # if len(h) == 2 then sample from K with probability prob_table[0,:]
                                k = np.random.choice(K, p=prob_table[len(h) - 2, :])

                                pos = h.index((gender, age))
                                susceptibles_ages = [x[1] for idx, x in enumerate(h) if idx != pos]
                                new_infected = np.random.choice(susceptibles_ages, size=k, replace=False)

                                for new_one in new_infected:
                                    output[i, new_one] += 1
        print(f'Check: Index cases count: {check_index_cases_count + check_index_cases_count_below90}')
        print(f'Index cases from db: {sum(data.number_of_cases)}')
        print(f'Number of people < 90 from db: {data[data.age < cutoff_age].number_of_cases.sum()}, '
              f'check: {check_index_cases_count_below90}')

        self.dump_pickles(output, 'sampled_households_secondary')
        self.dump_pickles(wielkosc_domkow, 'sanity_check_household_size')

        return output, wielkosc_domkow

    def get_infected_confidence_interval(self, sampled_households, age_groups=(40, 60, 80)):

        def get_confidence_interval(ag, alpha=0.01):
            s = sorted(np.sum(ag, axis=0))
            a = int(alpha * len(s) / 2)
            return s[a], s[len(s) - a]

        people_by_age = sampled_households.T
        confidence_intervals = []

        for i, age in enumerate(age_groups):
            start_idx = 0 if i == 0 else age_groups[i - 1]
            confidence_intervals.append(get_confidence_interval(people_by_age[start_idx:age]))
        confidence_intervals.append(get_confidence_interval(people_by_age[age_groups[-1]:]))

        self.dump_pickles(confidence_intervals, 'infected_confidence_intervals')
        return confidence_intervals

    @staticmethod
    def get_means(sampled_households, age_groups=(40, 60, 80)):
        people_by_age = sampled_households.T
        means = []

        for i, age in enumerate(age_groups):
            start_idx = 0 if i == 0 else age_groups[i - 1]
            means.append(np.mean(np.sum(people_by_age[start_idx:age], axis=0)))
        means.append(np.mean(np.sum(people_by_age[age_groups[-1]:], axis=0)))
        return means

    def plot_g_function(self, g_values):
        plot_path = str(RESULTS_DIR / f'G_of_lambda_{self.dt}.png')
        plot_g(g_values, plot_path)


def get_g_function_full_range():
    calc = AttackRate()
    index_cases = get_patient_data()
    index_cases_grouped_by_age = get_index_cases_grouped_by_age(index_cases)
    elderly_grouped = get_elderly_patient_data(index_cases)
    mu_k_lambda = calc.get_mu_k_lambda(full_range=True)
    lambdas = list(mu_k_lambda.columns)
    p, num_cases = calc.calculate_probabilities_of_household_size(index_cases_grouped_by_age, elderly_grouped)
    mu_bar = calc.get_mu_bar(p, mu_k_lambda, lambdas)
    ei_df = calc.get_ei(mu_bar, num_cases, lambdas)
    bar_h = calc.get_bar_h(p)
    EN_asterisk = calc.get_EN_asterisk(bar_h, num_cases)
    g_df = calc.get_g_function(ei_df, EN_asterisk)
    calc.plot_g_function(g_df)
    return g_df


def attack_rate_calculations(patient_data_file):
    calc = AttackRate()
    index_cases = get_patient_data(patient_data_path=patient_data_file)
    print(f'Number of index cases: {len(index_cases.index)}')
    index_cases_grouped_by_age = get_index_cases_grouped_by_age(index_cases)
    elderly_grouped = get_elderly_patient_data(index_cases)
    mu_k_lambda = calc.get_mu_k_lambda()
    lambdas = list(mu_k_lambda.columns)
    p, num_cases = calc.calculate_probabilities_of_household_size(index_cases_grouped_by_age, elderly_grouped)
    mu_bar = calc.get_mu_bar(p, mu_k_lambda, lambdas)
    ei_df = calc.get_ei(mu_bar, num_cases, lambdas)
    bar_h = calc.get_bar_h(p)
    EN_asterisk = calc.get_EN_asterisk(bar_h, num_cases)
    print(f'Expected number of susceptibles {EN_asterisk}')
    g_df = calc.get_g_function(ei_df, EN_asterisk)
    i_hat = get_known_secondary_infected_count(index_cases)
    print(f'Observed number of secondary infected {i_hat}')
    lambda_hat = calc.get_lambda_hat(g_df, i_hat, EN_asterisk)
    print(f'Estimated lambda: {lambda_hat}')
    prob_table, expected_infected = calc.get_probabilities_of_infection(lambda_hat)
    sampled_households, wielkosci_domkow = calc.sample_households_and_infect(index_cases_grouped_by_age,
                                                                             elderly_grouped,
                                                                             prob_table)
    print(f'Number of drawn households {sum(wielkosci_domkow[0, :])} should be equal to '
          f'the number of index cases  {len(index_cases.index)}')
    conf_intervals = calc.get_infected_confidence_interval(sampled_households)
    print('Confidence intervals: ', conf_intervals)
    means = calc.get_means(sampled_households)
    print('Means of infected: ', means)
    print('Sum of means: ', sum(means))
    print('Known secondary infected (age-grouped)', get_known_secondary_infected_age_grouped(
        patient_data_path=patient_data_file))


def lambda_reverse_engineering(lambda_hat, num_trials):
    calc = AttackRate()
    index_cases = get_patient_data()
    print(f'Number of index cases: {len(index_cases.index)}')
    index_cases_grouped_by_age = get_index_cases_grouped_by_age(index_cases)
    elderly_grouped = get_elderly_patient_data(index_cases)
    prob_table, expected_infected = calc.get_probabilities_of_infection(lambda_hat)
    sampled_households, household_sizes = calc.sample_households_and_infect(index_cases_grouped_by_age, elderly_grouped,
                                                                            prob_table, num_trials=num_trials)
    print(f'Number of drawn households {sum(household_sizes[0, :])} should be equal to '
          f'the number of index cases  {len(index_cases.index)}')
    conf_intervals = calc.get_infected_confidence_interval(sampled_households)
    print('Confidence intervals: ', conf_intervals)
    means = calc.get_means(sampled_households)
    print('Means of infected: ', means)
    print('Sum of means: ', sum(means))
    print('Known secondary infected (age-grouped)', get_known_secondary_infected_age_grouped())


def get_mean_infected(infected):
    return np.mean(infected, 0)


def lambda_bisection(sampled_households, known_secondary_infected, age_ranges=(20, 40, 60, 80),
                     max_iterations=6):
    """
    Searches for a lambda (attack rate) that is either an upper or a lower bound for a unifrom case.
    :param sampled_households: matrix of susceptibles to attack with infection
    :param known_secondary_infected: number of known secondary infected in each age group
    :param max_iterations: maximum number of bisections to be performed
    :return: lambda bisection function
    """
    calc = AttackRate()
    known_secondary_infected_arr = np.array([known_secondary_infected[i] for i in range(len(age_ranges) + 1)])

    def lower(mean_bootstrapped_infected, known_secondary_infected_arr, lambda_lb, lambda_ub, current_lambda):
        """
        Bisection function for lower bound calculations
        :param mean_bootstrapped_infected:
        :param known_secondary_infected_arr:
        :param lambda_lb:
        :param lambda_ub:
        :param current_lambda:
        :return:
        """
        if all(mean_bootstrapped_infected < known_secondary_infected_arr):
            return current_lambda, lambda_ub
        return lambda_lb, current_lambda

    def upper(mean_bootstrapped_infected, known_secondary_infected_arr, lambda_lb, lambda_ub, current_lambda):
        """
        Bisection function for upper bound calculations
        :param mean_bootstrapped_infected:
        :param known_secondary_infected_arr:
        :param lambda_lb:
        :param lambda_ub:
        :param current_lambda:
        :return:
        """
        if all(mean_bootstrapped_infected > known_secondary_infected_arr):
            return lambda_lb, current_lambda
        return current_lambda, lambda_ub

    def inner(lambda_lb, lambda_ub, bound, iteration=0):
        """
        5. Dla każdej grupy wiekowej sprawdzamy, czy średnia wylosowanych I w tej grupie wiekowej jest
        mniejsza / większa od obserwowanej I w tej grupie wiekowej i w zależności od tego odpowiednio
        zwiększamy / zmniejszamy lambdę tej grupy wiekowej.
        :param lambda_lb: vector of lambda lower bound
        :param lambda_ub: vector of lambda upper bound
        :param bound: 'upper' or 'lower'
        :param iteration: iteration counter
        :return: vector of estimated lambdas
        """
        current_lambda = (lambda_lb + lambda_ub) / 2
        infected = calc.infect_susceptibles(sampled_households, current_lambda)
        mean_bootstrapped_infected = get_mean_infected(infected)
        if iteration == max_iterations:
            return current_lambda, mean_bootstrapped_infected

        if bound == 'lower':
            next_lambda_lb, next_lambda_ub = lower(mean_bootstrapped_infected, known_secondary_infected_arr,
                                                   lambda_lb, lambda_ub, current_lambda)
        elif bound == 'upper':
            next_lambda_lb, next_lambda_ub = upper(mean_bootstrapped_infected, known_secondary_infected_arr,
                                                   lambda_lb, lambda_ub, current_lambda)
        else:
            raise ValueError(f'Unknown bound {bound}')

        return inner(next_lambda_lb, next_lambda_ub, bound, iteration + 1)

    return inner


def print_results(known, lower_lambda, lower_bound_infected, upper_lambda, upper_bound_infected):
    print(f'Known secondary infected & {known[0]} & {known[1]} & {known[2]} & {known[3]} & {known[4]} \\\\')
    print(f'Lower bound $\\lambda$ &  {lower_lambda} \\\\')
    print(f'Lower bound infected & {"& ".join(str(m) for m in lower_bound_infected)} \\\\')
    print(f'Upper bound $\\lambda$ &  {upper_lambda} \\\\')
    print(f'Upper bound infected & {"& ".join(str(m) for m in upper_bound_infected)} \\\\')


def lambda_bounds(lambda_lb, lambda_ub, age_ranges=(20, 40, 60, 80)):
    sampled_households = utils.load_pickles(
        Path(r'D:\python\dark-figure\results\test10000\sampled_households5d_202008092146.pickle').resolve())
    known_secondary_infected = get_known_secondary_infected_age_grouped(age_ranges=age_ranges)[0]
    lambda_bisection_function = lambda_bisection(sampled_households, known_secondary_infected)
    lower_lambda, lower_mean_infected = lambda_bisection_function(lambda_lb, lambda_ub, 'lower')
    utils.dump_pickles(lower_lambda, 'lower_bound', 'lambdas')
    utils.dump_pickles(lower_mean_infected, 'lower_bound', 'mean_infected')
    upper_lambda, upper_mean_infected = lambda_bisection_function(lambda_lb, lambda_ub, 'lower')
    utils.dump_pickles(upper_lambda, 'upper_bound', 'lambdas')
    utils.dump_pickles(upper_mean_infected, 'upper_bound', 'mean_infected')

    print_results(known_secondary_infected, lower_lambda, lower_mean_infected, upper_lambda, upper_mean_infected)
    return lower_lambda, upper_lambda


if __name__ == '__main__':
    """calc = AttackRate()
    sampled_households = calc.load_pickles(RESULTS_DIR / 'sampled_households_secondary_202007240956.pickle')
    conf_intervals = calc.get_infected_confidence_interval(sampled_households, age_groups=(20,40,60,80))
    print('Confidence intervals: ', conf_intervals)
    means = calc.get_means(sampled_households, age_groups=(20,40,60,80))
    print('Means of infected: ', means)"""
    # get_g_function_full_range()
    lambda_bounds(0.05, 0.15)
