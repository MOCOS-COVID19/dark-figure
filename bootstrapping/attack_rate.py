import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from bootstrapping.nsp2011 import get_households_within_habitants_range, get_data, get_data_for_voy, \
    get_people_in_households
from bootstrapping.infected_dataset import get_elderly_patient_data, get_patient_data, get_index_cases_grouped_by_age
from bootstrapping.settings import *


def get_K(max_household_size=DEFAULT_MAX_MIN_HOUSEHOLD_SIZE):
    return np.array(list(range(max_household_size - 1)))


def get_lambdas():
    return np.logspace(-10, -2, base=2, num=50)  # more dense at the beginning 0 to 0.15


def get_mu_k_lambda():
    mu_k_lambda = pd.read_csv(str(data_dir / 'mu_of_alpha.csv'), index_col=[0])
    mu_k_lambda.columns.name = 'lambda'
    return mu_k_lambda


def calculate_probabilities_of_household_size(data, elderly_data, household_size_min=1,
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
                                            ['age', 'number_of_cases']].set_index('age')['number_of_cases'].to_dict()
                for age, number_of_cases in sources_dict.items():
                    num_cases[voy_idx, age - age_min, gender, n - household_size_min] = number_of_cases
                    household_indices = random.choices(people[(gender, age)], k=num_trials)
                    for index in household_indices:
                        occurrences[voy_idx,
                                    age - age_min,
                                    gender,
                                    n - household_size_min,
                                    len(households[index]) - household_size_min] += 1

    return occurrences / num_trials, num_cases


def attack_rate_calculations():
    index_cases = get_patient_data()
    index_cases_grouped_by_age = get_index_cases_grouped_by_age(index_cases)
    elderly_grouped = get_elderly_patient_data(index_cases)
    lambdas = get_lambdas()
    mu_k_lambda = get_mu_k_lambda()
    K = get_K()
    p, num_cases = calculate_probabilities_of_household_size(index_cases_grouped_by_age, elderly_grouped)


def get_probability_of_getting_infected():
    prob_infection = pd.read_csv(str(data_dir / 'probability_of_infection.csv'))
    prob_infection = prob_infection.fillna(0)
    prob_infection = prob_infection.rename(columns={'k': 'household_size'})
    prob_infection.columns = [col.strip() for col in prob_infection.columns]
    prob_infection = prob_infection[
        ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13']]
    probability_table = prob_infection.to_numpy()
    return probability_table


def sample_household(data, elderly_data, prob_table, household_size_max=15, num_trials=10000, cutoff_age=90):
    all_people, all_households = get_data()

    output = np.zeros((num_trials, max(all_people.keys()) + 1))

    K = np.arange(0, 14)
    wielkosc_domkow = np.zeros((num_trials, len(K)))

    check_index_cases_count = 0
    for n in elderly_data.min_household_size.unique():

        households = get_households_within_habitants_range(all_households, n, household_size_max)
        people = get_people_in_households(all_people, households)

        for j, row in elderly_data[elderly_data.min_household_size == n].iterrows():
            for i in range(num_trials):
                age = row.age
                voyvodships = row.voy
                number_of_cases = len(voyvodships)
                check_index_cases_count += number_of_cases
                try:
                    household_indices = random.choices(people[age], k=number_of_cases)
                    assert number_of_cases == len(household_indices)
                except IndexError:
                    print('Household sampling blew up', age, number_of_cases)
                    continue

                for household_index in household_indices:
                    h = households[household_index]
                    wielkosc_domkow[i, len(h) - 1] += 1
                    if len(h) == 1:
                        continue
                    # if len(h) == 2 then sample from K with probability prob_table[0,:]
                    k = np.random.choice(K, p=prob_table[len(h) - 2, :])

                    pos = h.index(age)
                    new_infected = np.random.choice(h[:pos] + h[pos + 1:], size=k, replace=False)

                    for new_one in new_infected:
                        output[i, new_one] += 1
    print(f'Check: Index cases count: {check_index_cases_count}')
    # print(f'Index cases from db: {len(elderly_data.index)}')

    check_index_cases_count_below90 = 0
    for voy_idx, voy in enumerate(voy_mapping.values()):
        all_people, all_households = get_data_for_voy(voy)

        voy_data = data[(data.voy == voy) & (data.age < cutoff_age)]
        # check_index_cases_count_below90 += len(voy_data.index)

        for n in voy_data.min_household_size.unique():
            # print(f'n={n}')
            households = get_households_within_habitants_range(all_households, n, household_size_max)
            people = get_people_in_households(all_people, households)

            sources_dict = voy_data[voy_data.min_household_size == n][['age', 'number_of_cases']] \
                .set_index('age')['number_of_cases'].to_dict()
            # print(voy_data.number_of_cases.sum())
            check_index_cases_count_below90 += sum(sources_dict.values())

            for age, number_of_cases in sources_dict.items():
                for i in range(num_trials):

                    household_indices = random.choices(people[age], k=number_of_cases)
                    for household_index in household_indices:
                        h = households[household_index]
                        wielkosc_domkow[i, len(h) - 1] += 1
                        if len(h) == 1:
                            continue
                        # if len(h) == 2 then sample from K with probability prob_table[0,:]
                        k = np.random.choice(K, p=prob_table[len(h) - 2, :])

                        pos = h.index(age)
                        new_infected = np.random.choice(h[:pos] + h[pos + 1:], size=k, replace=False)

                        for new_one in new_infected:
                            output[i, new_one] += 1
    print(f'Check: Index cases count: {check_index_cases_count + check_index_cases_count_below90}')
    print(f'Index cases from db: {sum(data.number_of_cases)}')
    print(f'Number of people < 90 from db: {data[data.age < cutoff_age].number_of_cases.sum()}, '
          f'check: {check_index_cases_count_below90}')

    return output, wielkosc_domkow


"""main_num_trials = 10000
sampled_households, wielkosci_domkow = sample_household(get_index_cases_grouped_by_age, elderly_grouped,
                                                        probability_table,
                                                        num_trials=main_num_trials,
                                                        household_size_max=DEFAULT_MAX_MIN_HOUSEHOLD_SIZE)

dt = datetime.now().strftime('%Y%m%d%H%M')
with open(str(data_dir / f'sampled_households_secondary_{main_num_trials}_{dt}_olddata.pickle'), 'wb') as handle:
    pickle.dump(sampled_households, handle)

with open(str(data_dir / f'wielkosci_domkow_{main_num_trials}_{dt}.pickle'), 'wb') as handle:
    pickle.dump(wielkosci_domkow, handle)

print('completed')
"""

if __name__ == '__main__':
    attack_rate_calculations()
