import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import random
from .nsp2011 import get_households_within_habitants_range, get_data, get_data_for_voy, \
    get_people_in_households
from .infected_dataset import get_elderly_patient_data

project_dir = Path(__file__).resolve().parents[2]
notebooks_path = project_dir / 'notebooks'

index_cases = pd.read_csv(str(project_dir / 'addresses_ages_rev1_20200701.csv'))
print(f'Number of index cases without given age: {len(index_cases[index_cases.earliest_age_in_address.isna()].index)}')

index_cases = index_cases[~index_cases.earliest_age_in_address.isna()]
print(f'Number of index cases from DPS: {len(index_cases[index_cases.is_dps].index)}')

index_cases = index_cases[~index_cases.is_dps]
print(
    f'Number of index cases from large households: '
    f'{len(index_cases[index_cases.count_contacts_of_the_same_address >= 15].index)}')

index_cases = index_cases[index_cases.count_contacts_of_the_same_address < 15]

print(f'Number of index cases after NaN-age, large households and DPS removal: {len(index_cases.index)}')

voy_mapping = {'DOLNOŚLĄSKIE': '02', 'KUJAWSKO-POMORSKIE': '04', 'ŁÓDZKIE': '10', 'LUBELSKIE': '06',
               'LUBUSKIE': '08', 'MAŁOPOLSKIE': '12', 'MAZOWIECKIE': '14', 'OPOLSKIE': '16',
               'PODKARPACKIE': '18', 'PODLASKIE': '20', 'POMORSKIE': '22', 'ŚLĄSKIE': '24', 'ŚWIĘTOKRZYSKIE': '26',
               'WARMIŃSKO-MAZURSKIE': '28', 'WIELKOPOLSKIE': '30', 'ZACHODNIOPOMORSKIE': '32'}

index_cases = index_cases.replace({'Województwo': voy_mapping})

index_cases_grouped_by_age = index_cases \
    .groupby(by=['earliest_age_in_address', 'Województwo', 'count_contacts_of_the_same_address']) \
    .size().reset_index() \
    .rename(columns={
        0: 'number_of_cases',
        'earliest_age_in_address': 'age',
        'Województwo': 'voy',
        'count_contacts_of_the_same_address': 'min_household_size'})

index_cases_grouped_by_age.age = index_cases_grouped_by_age.age.astype(int)

print(index_cases_grouped_by_age['number_of_cases'].sum())

print(index_cases_grouped_by_age)

age_min = index_cases_grouped_by_age.age.min()

age_max = index_cases_grouped_by_age.age.max() + 1

print(f'Youngest index case: {age_min} years old, oldest index case: {age_max - 1} years old')

# K
main_household_size_max = 15

K = np.array(list(range(main_household_size_max - 1)))
print(K)

# Elderly

elderly_grouped = get_elderly_patient_data(index_cases)

# Second part

# Probability of getting infected
prob_infection = pd.read_csv(str(project_dir / 'probability_of_infection.csv'))
prob_infection = prob_infection.fillna(0)
prob_infection = prob_infection.rename(columns={'k': 'household_size'})
prob_infection.columns = [col.strip() for col in prob_infection.columns]
prob_infection = prob_infection[
    ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13']]
probability_table = prob_infection.to_numpy()


# Sampling households

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


main_num_trials = 10000
sampled_households, wielkosci_domkow = sample_household(index_cases_grouped_by_age, elderly_grouped, probability_table,
                                                        num_trials=main_num_trials,
                                                        household_size_max=main_household_size_max)

dt = datetime.now().strftime('%Y%m%d%H%M')
with open(str(project_dir / f'sampled_households_secondary_{main_num_trials}_{dt}_olddata.pickle'), 'wb') as handle:
    pickle.dump(sampled_households, handle)

with open(str(project_dir / f'wielkosci_domkow_{main_num_trials}_{dt}.pickle'), 'wb') as handle:
    pickle.dump(wielkosci_domkow, handle)

print('completed')
