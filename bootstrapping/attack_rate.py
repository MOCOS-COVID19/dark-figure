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
from bootstrapping.settings import *
from operator import itemgetter


class AttackRate:

    def __init__(self, max_household_size=DEFAULT_MAX_MIN_HOUSEHOLD_SIZE) -> None:
        self.dt = datetime.now().strftime('%Y%m%d%H%M')
        self.pickle_file_pattern = f'{{}}_{self.dt}.pickle'
        self.max_household_size = max_household_size
        self.K = self.get_K()
        self.lambdas = self.get_lambdas()

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

    @staticmethod
    def get_lambdas():
        return np.logspace(-10, -2, base=2, num=50)  # more dense at the beginning 0 to 0.15

    @staticmethod
    def get_mu_k_lambda():
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

    def get_mu_bar(self, probabilities, mu_k_lambda):
        # mu_k_lambda - dataframe, lambdas x household size (K-1)
        # probabilities - matrix (voy, age, gender, min household size, household size)
        lambdas_count = len(self.lambdas)
        voy_count = probabilities.shape[0]
        age_count = probabilities.shape[1]
        gender_count = probabilities.shape[2]
        household_size_count = probabilities.shape[3]

        mu_bar = np.zeros((lambdas_count, voy_count, age_count, gender_count, household_size_count))

        for lambda_idx, _lambda in enumerate(self.lambdas):
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
        def compute_number_of_infected(alpha, k, rng, x0=1):
            infected = np.zeros(k).astype(bool)
            sampled = np.zeros(k).astype(bool)
            infected[:x0] = 1
            while np.any(infected & ~sampled):
                arr = infected & ~sampled
                infecting_id = arr.nonzero()[0][0]
                sampled[infecting_id] = True
                probs = np.ones(k)
                probs[infecting_id] = 0
                probs /= (k - 1)
                infections = rng.binomial(k - 1, alpha)
                choices = rng.choice(k, infections, replace=False, p=probs)
                infected[choices] = True
            return sum(infected) - 1

        iterations = 20000
        expected = []
        k_values = np.arange(2, 15)  # this is the size of household
        inf_table = np.zeros((len(k_values), len(k_values) + 1))
        for k in k_values:
            expected_infected = 0
            for iters in np.arange(iterations):
                rng = np.random.default_rng()
                infected = compute_number_of_infected(_lambda, k, rng=rng)
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

    def get_ei(self, mu_bar, num_cases):
        EI = dict()
        for idx, _lambda in enumerate(self.lambdas):
            EI[_lambda] = np.sum(np.multiply(mu_bar[idx, :, :, :, :], num_cases))
        ei_df = pd.Series(data=EI).reset_index().rename(columns={'index': 'lambda', 0: 'EI'})
        self.dump_pickles(ei_df, 'ei_df')
        return ei_df

    def sample_household(self, data, elderly_data, prob_table, household_size_max=15, num_trials=10000, cutoff_age=90):
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
            start_idx = 0 if i == 0 else age_groups[i-1]
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


def attack_rate_calculations(patient_data_file):

    calc = AttackRate()
    index_cases = get_patient_data(patient_data_path=patient_data_file)
    print(f'Number of index cases: {len(index_cases.index)}')
    index_cases_grouped_by_age = get_index_cases_grouped_by_age(index_cases)
    elderly_grouped = get_elderly_patient_data(index_cases)
    mu_k_lambda = calc.get_mu_k_lambda()
    p, num_cases = calc.calculate_probabilities_of_household_size(index_cases_grouped_by_age, elderly_grouped)
    mu_bar = calc.get_mu_bar(p, mu_k_lambda)
    ei_df = calc.get_ei(mu_bar, num_cases)
    bar_h = calc.get_bar_h(p)
    EN_asterisk = calc.get_EN_asterisk(bar_h, num_cases)
    print(f'Expected number of susceptibles {EN_asterisk}')
    g_df = calc.get_g_function(ei_df, EN_asterisk)
    i_hat = get_known_secondary_infected_count(index_cases)
    print(f'Observed number of secondary infected {i_hat}')
    lambda_hat = calc.get_lambda_hat(g_df, i_hat, EN_asterisk)
    print(f'Estimated lambda: {lambda_hat}')
    prob_table, expected_infected = calc.get_probabilities_of_infection(lambda_hat)
    sampled_households, wielkosci_domkow = calc.sample_household(
        index_cases_grouped_by_age,
        elderly_grouped,
        prob_table)
    print(f'Number of drawn households {sum(wielkosci_domkow[0,:])} should be equal to '
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
    sampled_households, wielkosci_domkow = calc.sample_household(
        index_cases_grouped_by_age,
        elderly_grouped,
        prob_table, num_trials=num_trials)
    print(f'Number of drawn households {sum(wielkosci_domkow[0, :])} should be equal to '
          f'the number of index cases  {len(index_cases.index)}')
    conf_intervals = calc.get_infected_confidence_interval(sampled_households)
    print('Confidence intervals: ', conf_intervals)
    means = calc.get_means(sampled_households)
    print('Means of infected: ', means)
    print('Sum of means: ', sum(means))
    print('Known secondary infected (age-grouped)', get_known_secondary_infected_age_grouped())


if __name__ == '__main__':
    calc = AttackRate()
    sampled_households = calc.load_pickles(RESULTS_DIR / 'sampled_households_secondary_202007240956.pickle')
    conf_intervals = calc.get_infected_confidence_interval(sampled_households, age_groups=(20,40,60,80))
    print('Confidence intervals: ', conf_intervals)
    means = calc.get_means(sampled_households, age_groups=(20,40,60,80))
    print('Means of infected: ', means)


