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


