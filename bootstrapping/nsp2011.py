import pickle
from collections import defaultdict

import pandas as pd

from .settings import data_dir


def _recalculate_data(nsp_folder):
    all_people = defaultdict(list)
    all_households = defaultdict(list)

    for folder in nsp_folder.iterdir():
        population_voy = pd.read_csv(str(folder / 'population.csv'))
        for idx, row in population_voy.iterrows():
            all_people[(row.gender, row.age)].append(row.household_index)
            all_households[row.household_index].append((row.gender, row.age))

    with (data_dir / 'all_people.pickle').open('wb') as handle:
        pickle.dump(all_people, handle)

    with (data_dir / 'all_households.pickle').open('wb') as handle:
        pickle.dump(all_households, handle)
    return all_people, all_households


def get_data(recalculate=False, nsp_folder = data_dir / 'nsp'):

    if recalculate:
        return _recalculate_data(nsp_folder)
    try:
        with (data_dir / 'all_people.pickle').open('rb') as handle:
            all_people = pickle.load(handle)

        with (data_dir / 'all_households.pickle').open('rb') as handle:
            all_households = pickle.load(handle)

        return all_people, all_households

    except FileNotFoundError:
        return _recalculate_data(nsp_folder)




def get_data_for_voy(voy):
    import pickle

    if voy == '20':  # PODLASKIE
        return get_data()

    try:
        with (data_dir / f'all_people_{voy}.pickle').open('rb') as handle:
            all_people = pickle.load(handle)

        with (data_dir / f'all_households_{voy}.pickle').open('rb') as handle:
            all_households = pickle.load(handle)

    except FileNotFoundError:
        all_people = defaultdict(list)
        all_households = defaultdict(list)

        folder = data_dir / 'nsp' / voy

        population_voy = pd.read_csv(str(folder / 'population.csv'))
        for idx, row in population_voy.iterrows():
            all_people[(row.gender, row.age)].append(row.household_index)
            all_households[row.household_index].append((row.gender, row.age))

        with (data_dir / f'all_people_{voy}.pickle').open('wb') as handle:
            pickle.dump(all_people, handle)

        with (data_dir / f'all_households_{voy}.pickle').open('wb') as handle:
            pickle.dump(all_households, handle)
    return all_people, all_households


def get_households_within_habitants_range(households, min_n, max_n):
    """
    :param min_n - minimum household size inclusive
    :param max_n - maximum household size exclusive
    """
    return {key: val for key, val in households.items() if min_n <= len(val) < max_n}


def get_households_with_at_least_n_habitants(households, n):
    return {key: val for key, val in households.items() if len(val) >= n}


def get_households_with_less_than_n_inhabitants(households, n):
    return {key: val for key, val in households.items() if len(val) < n}


def get_people_in_households(all_people, households):
    people = {}
    for key, val in all_people.items():
        people[key] = [v for v in val if v in households.keys()]
    return people
