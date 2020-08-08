import pandas as pd
from bootstrapping.settings import *
from collections import defaultdict
import ast
from bisect import bisect_right


def string_to_list_len(x):
    return len([aa for aa in x[1:-1].split(', ') if len(aa) > 0])


def read_clean_patient_data(patient_data_path=str(data_dir / 'addresses_ages_rev2_20200701.csv'),
                            max_min_household_size=DEFAULT_MAX_MIN_HOUSEHOLD_SIZE):
    patient_data = pd.read_csv(patient_data_path)
    patient_data = patient_data[~patient_data.is_dps]
    patient_data = patient_data[
        (~patient_data.earliest_age_in_address.isna()) & (~patient_data.index_gender_in_address.isna())]
    patient_data.loc[(patient_data.earliest_age_in_address < 18) & (
            patient_data.count_contacts_of_the_same_address == 1), 'count_contacts_of_the_same_address'] = 2
    patient_data = patient_data[patient_data.count_contacts_of_the_same_address < max_min_household_size]
    patient_data = patient_data.replace({'Województwo': voy_mapping, 'index_gender_in_address': gender_mapping})
    patient_data.earliest_age_in_address = patient_data.earliest_age_in_address.astype(int)
    patient_data.index_gender_in_address = patient_data.index_gender_in_address.astype(int)
    return patient_data


def get_patient_data(patient_data_path=str(data_dir / 'addresses_ages_rev2_20200701.csv'),
                     max_min_household_size=DEFAULT_MAX_MIN_HOUSEHOLD_SIZE):
    patient_data = read_clean_patient_data(patient_data_path, max_min_household_size)
    patient_data = patient_data.rename(
        columns={'earliest_age_in_address': 'age',
                 'count_contacts_of_the_same_address': 'min_household_size',
                 'index_gender_in_address': 'gender',
                 'Województwo': 'voy'})
    patient_data = patient_data[['age', 'min_household_size', 'gender', 'voy']]
    return patient_data


def get_known_secondary_infected_count(patient_data):
    return patient_data.min_household_size.sum() - len(patient_data.index)


def get_count_by_field(fields, patient_data_path=str(data_dir / 'addresses_ages_rev2_20200701.csv'),
                                             max_min_household_size=DEFAULT_MAX_MIN_HOUSEHOLD_SIZE,
                                             age_ranges=(40, 60, 80)):
    infected_by_age_group = defaultdict(int)
    unknown_age = 0
    patient_data = read_clean_patient_data(patient_data_path, max_min_household_size)
    for field in fields:
        for stuff in patient_data[field]:
            try:
                ages = ast.literal_eval(stuff)
            except ValueError:
                ages = []
                for token in stuff[1:-1].split(', '):
                    if token == 'nan':
                        unknown_age += 1
                    else:
                        ages.append(float(token))
            for current_age in ages:
                group_id = bisect_right(age_ranges, current_age)
                infected_by_age_group[group_id] += 1

    return infected_by_age_group, unknown_age


def get_known_secondary_infected_age_grouped(patient_data_path=str(data_dir / 'addresses_ages_rev2_20200701.csv'),
                                             max_min_household_size=DEFAULT_MAX_MIN_HOUSEHOLD_SIZE,
                                             age_ranges=(40, 60, 80)):
    return get_count_by_field(['later_ages_in_address'], patient_data_path, max_min_household_size, age_ranges)


def get_index_cases_grouped_by_age(index_cases):
    grouped_cases = index_cases \
        .groupby(by=['age', 'gender', 'voy', 'min_household_size']) \
        .size().reset_index() \
        .rename(columns={0: 'number_of_cases'})

    grouped_cases.age = grouped_cases.age.astype(int)
    return grouped_cases


def get_elderly_patient_data(index_cases, cutoff_age=90):
    if 'earliest_age_in_address' in index_cases.columns:
        df_elderly = index_cases.loc[
            index_cases.earliest_age_in_address >= cutoff_age,
            ['earliest_age_in_address', 'count_contacts_of_the_same_address',
             'index_gender_in_address', 'Województwo']]

        elderly_grouped = df_elderly.groupby(by=['earliest_age_in_address', 'count_contacts_of_the_same_address',
                                                 'index_gender_in_address'])['Województwo'] \
            .apply(list).reset_index()

        elderly_grouped = elderly_grouped.rename(columns={'earliest_age_in_address': 'age',
                                                          'count_contacts_of_the_same_address': 'min_household_size',
                                                          'index_gender_in_address': 'gender',
                                                          'Województwo': 'voy'})
    else:
        df_elderly = index_cases.loc[
            index_cases.age >= cutoff_age,
            ['age', 'min_household_size', 'gender', 'voy']]

        elderly_grouped = df_elderly.groupby(by=['age', 'min_household_size', 'gender'])['voy'] \
            .apply(list).reset_index()

    elderly_grouped.age = elderly_grouped.age.astype(int)
    return elderly_grouped


def get_severe_10_age_grouped(patient_data_path=str(data_dir / 'addresses_ages_rev2_20200701.csv'),
                  max_min_household_size=DEFAULT_MAX_MIN_HOUSEHOLD_SIZE, age_ranges=(40,60,80)):
    return get_count_by_field(['later_ages_in_address_severe10', 'later_ages_in_address_deaths'],
                              patient_data_path, max_min_household_size, age_ranges)


def get_severe_14_age_grouped(patient_data_path=str(data_dir / 'addresses_ages_rev2_20200701.csv'),
                  max_min_household_size=DEFAULT_MAX_MIN_HOUSEHOLD_SIZE, age_ranges=(40,60,80)):
    return get_count_by_field(['later_ages_in_address_severe14', 'later_ages_in_address_deaths'],
                              patient_data_path, max_min_household_size, age_ranges)


if __name__ == '__main__':
    df = read_clean_patient_data()
    print(df.columns)
