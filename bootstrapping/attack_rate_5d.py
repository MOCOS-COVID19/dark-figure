import random
from bootstrapping.infection_prob_5d import InfectionProbabilitiesCalculator
from bisect import bisect_right

import numpy as np
from tqdm import tqdm

from bootstrapping.nsp2011 import get_households_within_habitants_range, get_data, get_data_for_voy, \
    get_people_in_households
from bootstrapping.settings import *
from bootstrapping.infected_dataset import (get_patient_data, get_elderly_patient_data, get_index_cases_grouped_by_age,
                                            get_known_secondary_infected_age_grouped)
from bootstrapping import utils


def _bootstrap(people, households, gender, age, number_of_cases, num_trials, sampled_households, age_ranges,
               index_case_index):
    for i in range(num_trials):
        household_indices = random.choices(people[(gender, age)], k=number_of_cases)
        for k, household_index in enumerate(household_indices):
            h = households[household_index]

            if len(h) > 1:
                pos = h.index((gender, age))
                for person_age in [x[1] for idx, x in enumerate(h) if idx != pos]:
                    sampled_households[i, index_case_index + k, bisect_right(age_ranges, person_age)] += 1


def _save_index_age_group(age, age_ranges, number_of_cases, index_cases_ages, index_case_index):
    # saving the age of index cases in the order of processing
    infected_age_group = bisect_right(age_ranges, age)
    for kk in range(number_of_cases):
        index_cases_ages[index_case_index + kk] = infected_age_group


def sample_households_for_index_cases(data, elderly_data, household_size_max=15, num_trials=10000, cutoff_age=90,
                                      age_ranges=(20, 40, 60, 80)):
    all_people, all_households = get_data()

    num_index_cases = data.number_of_cases.sum()
    num_age_groups = len(age_ranges) + 1
    sampled_households = np.zeros((num_trials, num_index_cases, num_age_groups))
    index_cases_ages = np.zeros(num_index_cases)

    index_case_index = 0
    for n in tqdm(elderly_data.min_household_size.unique()):

        households = get_households_within_habitants_range(all_households, n, household_size_max)
        people = get_people_in_households(all_people, households)

        for j, row in elderly_data[elderly_data.min_household_size == n].iterrows():
            age = row.age
            gender = row.gender
            voyvodships = row.voy
            number_of_cases = len(voyvodships)

            _save_index_age_group(age, age_ranges, number_of_cases, index_cases_ages, index_case_index)
            _bootstrap(people, households, gender, age, number_of_cases, num_trials, sampled_households,
                       age_ranges, index_case_index)
            index_case_index += number_of_cases

    indices = [68, 1227, 1617, 2411, 2732, 2790, 3560, 5011, 5280, 5583, 5718, 6054, 8293, 8489, 8615, 9564]

    for voy_idx, voy in enumerate(tqdm(voy_mapping.values())):
        all_people, all_households = get_data_for_voy(voy)

        voy_data = data[(data.voy == voy) & (data.age < cutoff_age)]
        assert index_case_index == indices[voy_idx], f'Expected {indices[voy_idx]} but got {index_case_index}'

        for n in voy_data.min_household_size.unique():
            households = get_households_within_habitants_range(all_households, n, household_size_max)
            people = get_people_in_households(all_people, households)

            for gender in GENDER_INDICES:
                sources_dict = voy_data.loc[(voy_data.gender == gender) & (voy_data.min_household_size == n),
                                            ['age', 'number_of_cases']] \
                    .set_index('age')['number_of_cases'].to_dict()

                for age, number_of_cases in sources_dict.items():
                    _save_index_age_group(age, age_ranges, number_of_cases, index_cases_ages, index_case_index)
                    _bootstrap(people, households, gender, age, number_of_cases, num_trials, sampled_households,
                               age_ranges, index_case_index)
                    index_case_index += number_of_cases

    return index_cases_ages, sampled_households


def infect(index_cases_ages, sampled_households, lambda_hat, prob_calc):
    lambda_hat = lambda_hat.tolist()
    sampled_households = sampled_households.astype(int)

    num_trials = sampled_households.shape[0]
    num_age_groups = sampled_households.shape[2]
    total_infected = np.zeros((num_trials, num_age_groups), dtype=int)
    for i in tqdm(range(num_trials)):

        for j, (age_group, household) in enumerate(zip(index_cases_ages, sampled_households[i])):
            probabilities = prob_calc.get_probabilities_of_infection(*lambda_hat, age_group, *household.tolist())

            non_zero_idx = np.where(household != 0)[0]
            for k in non_zero_idx:
                total_infected[i, k] += sum(np.random.rand(household[k]) < probabilities[k])

    return total_infected


def get_mean_infected(infected):
    return np.mean(infected, 0)


def lambda_bisection(index_cases_ages, sampled_households, known_secondary_infected, max_iterations=6):
    prob_calc = InfectionProbabilitiesCalculator()

    def inner(lambda_lb, lambda_ub, iteration=0):
        """
        5. Dla każdej grupy wiekowej sprawdzamy, czy średnia wylosowanych I/N w tej grupie wiekowej jest
        mniejsza / większa od obserwowanej I/N w tej grupie wiekowej i w zależności od tego odpowiednio
        zwiększamy / zmniejszamy lambdę tej grupy wiekowej.
        :param lambda_lb: vector of lambda lower bound
        :param lambda_ub: vector of lambda upper bound
        :param iteration: iteration counter
        :return: vector of estimated lambdas
        """
        current_lambda = (lambda_lb + lambda_ub) / 2  # this need to be in 5D
        infected = infect(index_cases_ages, sampled_households, current_lambda, prob_calc)
        mean_infected = get_mean_infected(infected)
        if iteration == max_iterations:
            return current_lambda, mean_infected

        next_lambda_lb = lambda_lb
        next_lambda_ub = lambda_ub
        for i, calculated in enumerate(mean_infected):
            if calculated < known_secondary_infected[i]:
                # increase lambda
                next_lambda_lb[i] = current_lambda[i]
            else:
                # decrease lambda
                next_lambda_ub[i] = current_lambda[i]
        return inner(next_lambda_lb, next_lambda_ub, iteration + 1)

    return inner


def print_results(known, lambdas, mean_infected):
    print(f'Known secondary infected | {known[0]} | {known[1]} | {known[2]} | {known[3]} | {known[4]} |')
    lambdas_str = [f'{l:.4f}' for l in lambdas]
    print(f'Estimated lambda | {"| ".join(lambdas_str)} |')
    print(f'Estimated mean | {"| ".join(str(m) for m in mean_infected)} |')


def main(subfolder, start_lambda_lb, start_lambda_ub, age_ranges = (20, 40, 60, 80), num_trials=10000):

    index_cases = get_patient_data()
    index_cases_grouped_by_age = get_index_cases_grouped_by_age(index_cases)
    elderly_grouped = get_elderly_patient_data(index_cases)
    index_cases_ages, sampled_households = sample_households_for_index_cases(
        index_cases_grouped_by_age,
        elderly_grouped,
        num_trials=num_trials, age_ranges=age_ranges)
    utils.dump_pickles(index_cases_ages, subfolder, 'index_cases_age_groups5d')
    utils.dump_pickles(sampled_households, subfolder, 'sampled_households5d')

    known_secondary_infected = get_known_secondary_infected_age_grouped(age_ranges=age_ranges)[0]
    lambda_bisection_function = lambda_bisection(index_cases_ages, sampled_households, known_secondary_infected)
    lambdas, mean_infected = lambda_bisection_function(start_lambda_lb, start_lambda_ub)
    utils.dump_pickles(lambdas, subfolder, 'lambda5d')
    utils.dump_pickles(mean_infected, subfolder, 'mean_infected5d')

    print_results(known_secondary_infected, lambdas, mean_infected)
    return lambdas


if __name__ == '__main__':
    # lb = 0.05 * np.ones((5,))
    #ub = 0.15 * np.ones((5,))
    #main('test10000', lb, ub)
    subfolder = 'test10000'
    current_lambda = utils.load_pickles(
        Path(r'D:\python\dark-figure\results\test10000\lambda5d_202008110214.pickle').resolve())
    index_cases_ages = utils.load_pickles(
        Path(r'D:\python\dark-figure\results\test10000\index_cases_age_groups5d_202008092146.pickle').resolve())
    sampled_households = utils.load_pickles(
        Path(r'D:\python\dark-figure\results\test10000\sampled_households5d_202008092146.pickle').resolve())
    prob_calc = InfectionProbabilitiesCalculator()
    infected = infect(index_cases_ages, sampled_households, current_lambda, prob_calc)
    mean_infected = get_mean_infected(infected)
    utils.dump_pickles(mean_infected, subfolder, 'mean_infected5d_v2')
    infected = utils.load_pickles(Path(r'D:\python\dark-figure\results\test10000\infected_v2_202008112109.pickle').resolve())


