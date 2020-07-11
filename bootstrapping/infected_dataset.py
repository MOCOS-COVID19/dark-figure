import pandas as pd
from bootstrapping.settings import *


def string_to_list_len(x):
    return len([aa for aa in x[1:-1].split(', ') if len(aa) > 0])


def get_patient_data(patient_data_path=str(data_dir / 'addresses_ages_rev2_20200701.csv'),
                     max_min_household_size=DEFAULT_MAX_MIN_HOUSEHOLD_SIZE):
    patient_data = pd.read_csv(patient_data_path)
    patient_data = patient_data[~patient_data.is_dps]

    patient_data = patient_data.rename(
        columns={'earliest_age_in_address': 'age',
                 'count_contacts_of_the_same_address': 'min_household_size',
                 'index_gender_in_address': 'gender',
                 'Województwo': 'voy'})
    patient_data = patient_data[['age', 'min_household_size', 'gender', 'voy']]
    patient_data = patient_data[(~patient_data.age.isna()) & (~patient_data.gender.isna())]
    patient_data.loc[(patient_data.age < 18) & (patient_data.min_household_size == 1), 'min_household_size'] = 2
    patient_data = patient_data[patient_data.min_household_size < max_min_household_size]
    patient_data = patient_data.replace({'voy': voy_mapping, 'gender': gender_mapping})
    patient_data.age = patient_data.age.astype(int)
    patient_data.gender = patient_data.gender.astype(int)
    return patient_data


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
                                                 'index_gender_in_address'])['Województwo']\
            .apply(list).reset_index()

        elderly_grouped = elderly_grouped.rename(columns={'earliest_age_in_address': 'age',
                                                          'count_contacts_of_the_same_address': 'min_household_size',
                                                          'index_gender_in_address': 'gender',
                                                          'Województwo': 'voy'})
    else:
        df_elderly = index_cases.loc[
            index_cases.age >= cutoff_age,
            ['age', 'min_household_size', 'gender', 'voy']]

        elderly_grouped = df_elderly.groupby(by=['age', 'min_household_size', 'gender'])['voy']\
            .apply(list).reset_index()

    elderly_grouped.age = elderly_grouped.age.astype(int)
    return elderly_grouped


if __name__ == '__main__':
    df = get_patient_data()
    print(df.columns)
    print(df.head())
    df2 = get_elderly_patient_data(df)
    print(df2.columns)
    print(df2.head())
    df2['number_of_cases'] = df2['voy'].apply(lambda x: len(x))
    sources_dict = df2[df2.min_household_size == 2][['age', 'gender', 'number_of_cases']] \
        .set_index(['gender', 'age']).to_dict()['number_of_cases']
    for (gender, age), count in sources_dict.items():
        print(age, gender, count)
