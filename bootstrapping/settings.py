from pathlib import Path
data_dir = Path(r'd:\coronavirus\attack rate\data').resolve()
RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'

voy_mapping = {'DOLNOŚLĄSKIE': '02', 'KUJAWSKO-POMORSKIE': '04', 'ŁÓDZKIE': '10', 'LUBELSKIE': '06',
               'LUBUSKIE': '08', 'MAŁOPOLSKIE': '12', 'MAZOWIECKIE': '14', 'OPOLSKIE': '16',
               'PODKARPACKIE': '18', 'PODLASKIE': '20', 'POMORSKIE': '22', 'ŚLĄSKIE': '24', 'ŚWIĘTOKRZYSKIE': '26',
               'WARMIŃSKO-MAZURSKIE': '28', 'WIELKOPOLSKIE': '30', 'ZACHODNIOPOMORSKIE': '32'}
VOYVODSHIPS = ['02', '04', '06', '08', '10', '12', '14', '16', '18', '20', '22', '24', '26', '28', '30', '32']

DEFAULT_MAX_MIN_HOUSEHOLD_SIZE = 15
DEFAULT_NUM_TRIALS = 10000

MALE = 'Mężczyzna'
FEMALE = 'Kobieta'
MALE_IDX = 0
FEMALE_IDX = 1
GENDER_INDICES = [MALE_IDX, FEMALE_IDX]
gender_mapping = {MALE: MALE_IDX, FEMALE: FEMALE_IDX}

DEFAULT_CUTOFF_AGE = 90
