import bisect
import pickle
import random
from collections import defaultdict
from datetime import datetime

import numpy as np

from .infected_dataset import get_patient_data, voy_mapping, get_elderly_patient_data, \
    index_cases_grouped_by_age
from .nsp2011 import get_data_for_voy, get_data, get_people_in_households, \
    get_households_within_habitants_range


def decide_on_group_ages(list_of_ages, age_ranges, current_age):
    people_count = {key: 0 for key in range(len(age_ranges) + 1)}
    for person_age in list_of_ages:
        people_count[bisect.bisect_right(age_ranges, person_age)] += 1

    people_count[bisect.bisect_right(age_ranges, current_age)] -= 1
    return people_count


def calculate_household_sizes(sources_dict, age_ranges, all_people, all_households, members_count):
    for age, count in sources_dict.items():
        household_indices = random.choices(all_people[age], k=count)
        for household_idx in household_indices:
            household = all_households[household_idx]
            age_groups_count = decide_on_group_ages(household, age_ranges, age)
            for key, val in age_groups_count.items():
                members_count[key] += val
    return members_count


def zerodict(age_ranges):
    def create():
        return {key: 0 for key in range(len(age_ranges) + 1)}

    return create


def bootstrap_susceptibles(data, elderly_data, age_ranges, household_size_max=15, num_trials=10000, cutoff_age=90):
    full_result = defaultdict(zerodict(age_ranges))

    all_people, all_households = get_data()

    elderly_data['number_of_cases'] = elderly_data['voy'].apply(lambda x: len(x))

    for n in elderly_data.min_household_size.unique():

        households = get_households_within_habitants_range(all_households, n, household_size_max)
        people = get_people_in_households(all_people, households)

        sources_dict = elderly_data[elderly_data.min_household_size == n][['age', 'number_of_cases']]\
            .set_index('age').to_dict()['number_of_cases']

        for i in range(num_trials):
            calculate_household_sizes(sources_dict, age_ranges, people, households, full_result[i])

    for voy_idx, voy in enumerate(voy_mapping.values()):
        all_people, all_households = get_data_for_voy(voy)

        voy_data = data[(data.voy == voy) & (data.age < cutoff_age)]

        for n in voy_data.min_household_size.unique():
            households = get_households_within_habitants_range(all_households, n, household_size_max)
            people = get_people_in_households(all_people, households)

            sources_dict = voy_data[voy_data.min_household_size == n][['age', 'number_of_cases']] \
                .set_index('age')['number_of_cases'].to_dict()

            for i in range(num_trials):
                calculate_household_sizes(sources_dict, age_ranges, people, households, full_result[i])

    return full_result


def get_99percentile(results, num_trials):
    idx = int(np.min((np.ceil(0.99 * num_trials), num_trials - 1)))
    percentiles = {i: sorted(values)[idx] for i, values in results.items()}
    return percentiles


def get_means(results):
    return {i: np.mean(val) for i, val in results.items()}


def dict_to_lists(results):
    age_ranges_lists = defaultdict(list)
    for key, val in results.items():  # num_trial: dict of age_group: sum
        for key2, val2 in val.items():
            age_ranges_lists[key2].append(val2)
    return age_ranges_lists


def process_results_interim(interim_results, num_trials):
    return get_99percentile(interim_results, num_trials), get_means(interim_results)


def process_results(results, num_trials):
    interim_results = dict_to_lists(results)
    return process_results_interim(interim_results, num_trials)


def get_beta(y, ag_severe_cases, conf_level, severness_rate_lb, severness_rate_ub, epsilon):
    from scipy.stats import binom
    current_severness = (severness_rate_lb + severness_rate_ub) / 2
    print(f'Checking for {current_severness}')
    num_trials = len(y)  # 10000
    x = np.zeros((num_trials,))
    for i, y_i in enumerate(y):
        x[i] = binom.rvs(y_i, current_severness, loc=0, size=1, random_state=None)
    occurrences = np.count_nonzero(x >= ag_severe_cases)
    q = occurrences / num_trials
    print(f'q = {q}')
    # error within limit
    if 0 < q - conf_level < epsilon:
        return current_severness

    if q > conf_level:
        return get_beta(y, ag_severe_cases, conf_level, severness_rate_lb, current_severness, epsilon)
    return get_beta(y, ag_severe_cases, conf_level, current_severness, severness_rate_ub, epsilon)


def get_betas(age_groups_ids, y, severe_cases, conf_level, severness_rate_lb, severness_rate_ub, epsilon):
    betas = {}
    for age_group_id in age_groups_ids:
        print(f'Checking for age group {age_group_id}')
        ag_severe_cases = severe_cases[age_group_id]
        betas[age_group_id] = get_beta(y[age_group_id], ag_severe_cases, conf_level,
                                       severness_rate_lb, severness_rate_ub, epsilon)
    return betas


def get_alpha(y, num_trials, ag_severe_cases, conf_level, severness_rate_lb, severness_rate_ub, epsilon):
    from scipy.stats import binom
    current_severness = (severness_rate_lb + severness_rate_ub) / 2
    print(f'Checking for {current_severness}')
    print(f'num_trials = {num_trials}')  # 10000
    x = np.zeros((num_trials,))
    for i, y_i in enumerate([y] * num_trials):
        x[i] = binom.rvs(y_i, current_severness, loc=0, size=1, random_state=None)
    occurrences = np.count_nonzero(x < ag_severe_cases)
    q = occurrences / num_trials
    print(f'q = {q}')
    # error within limit
    if 0 < q - conf_level < epsilon:
        return current_severness
    if q < conf_level:
        return get_alpha(y, num_trials, ag_severe_cases, conf_level, severness_rate_lb, current_severness, epsilon)
    return get_alpha(y, num_trials, ag_severe_cases, conf_level, current_severness, severness_rate_ub, epsilon)


def get_alphas(age_groups_ids, observed_cases, num_trials, severe_cases, conf_level, severness_rate_lb,
               severness_rate_ub, epsilon):
    alphas = {}
    for age_group_id in age_groups_ids:
        print(f'Checking for age group {age_group_id}')
        ag_severe_cases = severe_cases[age_group_id]
        alphas[age_group_id] = get_alpha(observed_cases[age_group_id], num_trials, ag_severe_cases, conf_level,
                                         severness_rate_lb, severness_rate_ub, epsilon)
    return alphas


def dark_paper(num_trials=10000):
    age_ranges = [40, 60, 80]
    patient_data = get_patient_data()
    elderly_data = get_elderly_patient_data(patient_data)
    patient_data = index_cases_grouped_by_age(patient_data)
    results = bootstrap_susceptibles(patient_data, elderly_data, age_ranges, num_trials=num_trials)

    percentile99, mean = process_results(results, num_trials)
    print('Percentile 99')
    print(percentile99)
    print('Mean')
    print(mean)

    interim = dict_to_lists(results)
    dt = datetime.now().strftime('%Y%m%d%H%M')
    age_ranges_str = ''.join(str(x) for x in age_ranges)
    interim_file_name = f'{dt}_{num_trials}_{age_ranges_str}_susceptibles.pickle'

    print(interim_file_name)
    with open(interim_file_name, 'wb') as handle:
        pickle.dump(interim, handle)

    severe_cases_14 = {0: 45, 1: 84, 2: 103, 3: 45}
    severe_cases_10 = {0: 80, 1: 107, 2: 132, 3: 48}
    observed_cases = {0: 1807, 1: 1018, 2: 589, 3: 139}

    age_groups_ids = [0, 1, 2, 3]
    severness_rate_lb = 0.0
    severness_rate_ub = 0.1
    conf_level = 0.01
    epsilon = 0.001
    betas10 = get_betas(age_groups_ids, interim, severe_cases_10, conf_level, severness_rate_lb, severness_rate_ub,
                        epsilon)
    print('Beta 10-day')
    print(betas10)
    betas_file_name = f'{dt}_{num_trials}_{age_ranges_str}_betas10.pickle'
    print(betas_file_name)
    with open(betas_file_name, 'wb') as handle:
        pickle.dump(betas10, handle)

    betas14 = get_betas(age_groups_ids, interim, severe_cases_14, conf_level, severness_rate_lb, severness_rate_ub,
                        epsilon)
    print('Beta 14-day')
    print(betas14)
    betas_file_name = f'{dt}_{num_trials}_{age_ranges_str}_betas14.pickle'
    print(betas_file_name)
    with open(betas_file_name, 'wb') as handle:
        pickle.dump(betas14, handle)

    severness_rate_lb = 0.0
    severness_rate_ub = 0.8
    alphas_10 = get_alphas(age_groups_ids, observed_cases, num_trials, severe_cases_10, conf_level, severness_rate_lb,
                           severness_rate_ub, epsilon)
    print('Alpha 10-day')
    print(alphas_10)
    alphas_file_name = f'{dt}_{num_trials}_{age_ranges_str}_alphas10.pickle'
    with open(alphas_file_name, 'wb') as handle:
        pickle.dump(alphas_10, handle)

    alphas_14 = get_alphas(age_groups_ids, observed_cases, num_trials, severe_cases_14, conf_level, severness_rate_lb,
                           severness_rate_ub, epsilon)
    print('Alpha 14-day')
    print(alphas_14)
    alphas_file_name = f'{dt}_{num_trials}_{age_ranges_str}_alphas14.pickle'
    with open(alphas_file_name, 'wb') as handle:
        pickle.dump(alphas_14, handle)


def age_distribution(num_trials=10000):
    age_ranges = list(range(0, 101))
    patient_data = get_patient_data()
    elderly_data = get_elderly_patient_data(patient_data)
    patient_data = index_cases_grouped_by_age(patient_data)
    results = bootstrap_susceptibles(patient_data, elderly_data, age_ranges, num_trials=num_trials)

    interim = dict_to_lists(results)
    dt = datetime.now().strftime('%Y%m%d%H%M')
    age_ranges_str = ''.join(str(x) for x in age_ranges)
    interim_file_name = f'{dt}_{num_trials}_age_distribution.pickle'

    print(interim_file_name)
    with open(interim_file_name, 'wb') as handle:
        pickle.dump(interim, handle)


if __name__ == '__main__':
    dark_paper()
    age_distribution()
