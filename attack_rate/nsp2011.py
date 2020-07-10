from collections import defaultdict
from pathlib import Path
import pandas as pd


project_dir = Path(__file__).resolve().parents[2]


def get_data():
    import pickle

    try:
        with (project_dir / 'all_people.pickle').open('rb') as handle:
            all_people = pickle.load(handle)

        with (project_dir / 'all_households.pickle').open('rb') as handle:
            all_households = pickle.load(handle)

    except FileNotFoundError:
        all_people = defaultdict(list)
        all_households = defaultdict(list)

        nsp_folder = project_dir / 'data' / 'simulations' / 'nsp2011_powiats' / 'apartments'

        for folder in nsp_folder.iterdir():
            population_voy = pd.read_csv(str(folder / 'population.csv'))
            for idx, row in population_voy.iterrows():
                all_people[row.age].append(row.household_index)
                all_households[row.household_index].append(row.age)

        with (project_dir / 'all_people.pickle').open('wb') as handle:
            pickle.dump(all_people, handle)

        with (project_dir / 'all_households.pickle').open('wb') as handle:
            pickle.dump(all_households, handle)

    return all_people, all_households


def get_data_for_voy(voy):
    import pickle

    if voy == '20':  # PODLASKIE
        return get_data()

    try:
        with (project_dir / f'all_people_{voy}.pickle').open('rb') as handle:
            all_people = pickle.load(handle)

        with (project_dir / f'all_households_{voy}.pickle').open('rb') as handle:
            all_households = pickle.load(handle)

    except FileNotFoundError:
        all_people = defaultdict(list)
        all_households = defaultdict(list)

        folder = project_dir / 'data' / 'simulations' / 'nsp2011_powiats' / 'apartments' / voy

        population_voy = pd.read_csv(str(folder / 'population.csv'))
        for idx, row in population_voy.iterrows():
            all_people[row.age].append(row.household_index)
            all_households[row.household_index].append(row.age)

        with (project_dir / f'all_people_{voy}.pickle').open('wb') as handle:
            pickle.dump(all_people, handle)

        with (project_dir / f'all_households_{voy}.pickle').open('wb') as handle:
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

