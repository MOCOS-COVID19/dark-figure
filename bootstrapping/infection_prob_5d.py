import numpy as np


class InfectionProbabilitiesCalculator:
    def __init__(self, repeats=1000, rng=None):
        self.precalculated_values = dict()
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng
        self.repeats = repeats

    def _calculate_infection_probabilities(self, lambdas, index_age_group, household_size_age_grouped):

        probabilities = np.zeros(5)
        for i in range(5):
            if household_size_age_grouped[i] == 0:
                probabilities[i] = None
        k = int(np.sum(household_size_age_grouped))  # secondary household members
        for repeat in np.arange(self.repeats):

            # 1. preprocess and setup the job
            simulated = (0, 0, 0, 0, 0)
            infected = np.zeros(k).astype(bool)
            sampled = np.zeros(k).astype(bool)
            age_groups = np.zeros(k)
            start = 0
            for i, how_many in enumerate(household_size_age_grouped):
                stop = start + int(how_many)
                age_groups[start:stop] = i
                start += stop - start

            # 2. zero iteration - infect from source
            for i in range(5):
                group_size = household_size_age_grouped[i]
                if group_size == 0:
                    continue
                age_group_probs = np.zeros_like(age_groups)
                age_group_probs[age_groups == i] = 1 / group_size
                index_age_group_infections = self.rng.binomial(group_size, lambdas[i])
                choices = self.rng.choice(k, index_age_group_infections, replace=False, p=age_group_probs)
                infected[choices] = True
            # print(infected)
            # 3. remaining iterations - infect from infected
            while np.any(infected & ~sampled):
                arr = infected & ~sampled
                infecting_id = arr.nonzero()[0][0]
                sampled[infecting_id] = True

                age_group_of_infecting = age_groups[infecting_id]
                for i in range(5):
                    group_size = household_size_age_grouped[i]
                    if age_group_of_infecting == i:
                        group_size -= 1
                    if group_size == 0:
                        continue
                    age_group_probs = np.zeros_like(age_groups)
                    age_group_probs[age_groups == i] = 1 / group_size
                    age_group_probs[infecting_id] = 0

                    index_age_group_infections = self.rng.binomial(group_size, lambdas[i])
                    choices = self.rng.choice(k, index_age_group_infections, replace=False, p=age_group_probs)
                    infected[choices] = True
            # 4. update probabilities
            for i in range(5):
                if household_size_age_grouped[i] > 0:
                    age_group_infected = sum(infected[age_groups == i])
                    current_simulated_value = age_group_infected / household_size_age_grouped[i]
                    # online average
                    probabilities[i] = (repeat * probabilities[i] + current_simulated_value) / (repeat + 1)

        return probabilities

    def get_probabilities_of_infection(self, lambda1, lambda2, lambda3, lambda4, lambda5, index_age_group,
                                       household_size_age_group1, household_size_age_group2,
                                       household_size_age_group3, household_size_age_group4,
                                       household_size_age_group5):

        dict_key = ((lambda1, lambda2, lambda3, lambda4, lambda5), index_age_group,
                    (household_size_age_group1, household_size_age_group2, household_size_age_group3,
                     household_size_age_group4, household_size_age_group5))

        if dict_key not in self.precalculated_values:
            self.precalculated_values[dict_key] = self._calculate_infection_probabilities((lambda1, lambda2,
                                                                                           lambda3, lambda4,
                                                                                           lambda5), index_age_group,
                                                                                          (household_size_age_group1,
                                                                                           household_size_age_group2,
                                                                                           household_size_age_group3,
                                                                                           household_size_age_group4,
                                                                                           household_size_age_group5))
        return self.precalculated_values[dict_key]
