import random
from operator import itemgetter
from bisect import bisect_right

import numpy as np
from tqdm import tqdm

from bootstrapping.nsp2011 import get_households_within_habitants_range, get_data, get_data_for_voy, \
    get_people_in_households
from bootstrapping.settings import *
from bootstrapping.infected_dataset import (get_elderly_patient_data, get_patient_data, get_index_cases_grouped_by_age)
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


def infect(data, elderly_data, prob_table, household_size_max=15, num_trials=10000, cutoff_age=90):
    all_people, all_households = get_data()

    max_age_in_census = max(all_people.keys(), key=itemgetter(1))[1]
    output = np.zeros((num_trials, max_age_in_census + 1))

    K = np.arange(0, 14)  # self.K
    sampled_household_sizes = np.zeros((num_trials, len(K)))

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
                    sampled_household_sizes[i, len(h) - 1] += 1
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

    check_index_cases_count_below90 = 0
    for voy_idx, voy in enumerate(tqdm(voy_mapping.values())):
        all_people, all_households = get_data_for_voy(voy)

        voy_data = data[(data.voy == voy) & (data.age < cutoff_age)]

        for n in voy_data.min_household_size.unique():
            # print(f'n={n}')
            households = get_households_within_habitants_range(all_households, n, household_size_max)
            people = get_people_in_households(all_people, households)

            for gender in GENDER_INDICES:
                sources_dict = voy_data.loc[(voy_data.gender == gender) & (voy_data.min_household_size == n),
                                            ['age', 'number_of_cases']] \
                    .set_index('age')['number_of_cases'].to_dict()
                check_index_cases_count_below90 += sum(sources_dict.values())

                for age, number_of_cases in sources_dict.items():
                    for i in range(num_trials):

                        household_indices = random.choices(people[(gender, age)], k=number_of_cases)
                        for household_index in household_indices:
                            h = households[household_index]
                            sampled_household_sizes[i, len(h) - 1] += 1
                            if len(h) == 1:
                                continue
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

    return output, sampled_household_sizes


def lambda_bisection(index_cases_ages, sampled_households, max_iterations=6):

    def inner(lambda_lb, lambda_ub, iteration):
        current_lambda = (lambda_lb + lambda_ub) / 2  # this need to be in 5D
        if iteration == max_iterations:
            return current_lambda
        output = infect(index_cases_ages, sampled_households)

    return inner


def main(subfolder, start_lambda_lb, start_lambda_ub, num_trials=10000):
    """index_cases = get_patient_data()
    index_cases_grouped_by_age = get_index_cases_grouped_by_age(index_cases)
    elderly_grouped = get_elderly_patient_data(index_cases)
    index_cases_ages, sampled_households = sample_households_for_index_cases(
        index_cases_grouped_by_age,
        elderly_grouped,
        num_trials=num_trials)
    utils.dump_pickles(index_cases_ages, subfolder, 'index_cases_age_groups5d')
    utils.dump_pickles(sampled_households, subfolder, 'sampled_households5d')"""

    index_cases_ages = utils.load_pickles(
        Path(r'D:\python\dark-figure\results\test2\index_cases_age_groups5d_202008092133.pickle').resolve())
    sampled_households = utils.load_pickles(
        Path(r'D:\python\dark-figure\results\test2\sampled_households5d_202008092133.pickle').resolve())
    lambda_bisection_function = lambda_bisection(index_cases_ages, sampled_households)
    result = lambda_bisection_function(start_lambda_lb, start_lambda_ub, 0)
    utils.dump_pickles(result, subfolder, 'lambda5d')


if __name__ == '__main__':
    # main('test10000', num_trials=10000)
    lb = 0.05 * np.ones((5,))
    ub = 0.15 * np.ones((5,))
    main('test2', lb, ub, num_trials=2)
    # index_cases_ages = utils.load_pickles(Path(r'D:\python\dark-figure\results\test2\index_cases_age_groups5d_202008092133.pickle').resolve())
    # sampled_households = utils.load_pickles(Path(r'D:\python\dark-figure\results\test2\sampled_households5d_202008092133.pickle').resolve())

