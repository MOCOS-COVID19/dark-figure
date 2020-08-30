import numpy as np
from unittest import TestCase
from bootstrapping import attack_rate_5d


class TestAttackRate5d(TestCase):

    def setUp(self) -> None:
        self.prob_calc = attack_rate_5d.InfectionProbabilitiesCalculator()

    def test_infect(self):
        index_cases_ages = np.array([4, 3, 2, 0])
        num_trials = 6
        sampled_households = 5 * np.array([
            [[1, 0, 0, 0, 0], [0, 0, 0, 0, 2], [0, 1, 0, 1, 0], [1, 1, 1, 1, 1]],
            [[1, 0, 0, 0, 0], [0, 0, 0, 0, 2], [0, 1, 0, 1, 0], [1, 1, 1, 1, 1]],
            [[0, 1, 0, 0, 0], [0, 0, 0, 1, 1], [1, 0, 1, 0, 0], [1, 1, 1, 1, 1]],
            [[0, 0, 1, 0, 0], [0, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 1, 1, 1, 1]],
            [[0, 0, 0, 1, 0], [0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [1, 1, 1, 1, 1]],
            [[0, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [1, 1, 1, 1, 1]]
        ])

        lambdas = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

        res = attack_rate_5d.infect(index_cases_ages, sampled_households, lambdas, self.prob_calc,
                                    num_trials=num_trials)
        self.assertEqual(res.shape[0], sampled_households.shape[0])
        self.assertEqual(res.shape[1], sampled_households.shape[2])
        self.assertTrue(np.all(res <= np.sum(sampled_households, 1)))

        print(res)

        mean_infected = attack_rate_5d.get_mean_infected(res)
        self.assertEqual(5, len(mean_infected))

        print(mean_infected)

    def test_attack_rate_5d(self):
        from pathlib import Path
        from bootstrapping.attack_rate_5d import utils, InfectionProbabilitiesCalculator, infect, get_mean_infected
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
        utils.dump_pickles(infected, subfolder, 'infected_v2')


class TestAttackRate(TestCase):
    def test_quantiles(self):
        from bootstrapping.attack_rate import get_99_quantile_infected
        infected = np.repeat(np.arange(0, 1000), 5).reshape((1000, 5))
        result = get_99_quantile_infected(infected)
        self.assertEqual(5, len(result))
        self.assertEqual(990, result[0])
        self.assertEqual(990, result[3])

