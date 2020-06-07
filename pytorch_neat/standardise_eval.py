# Copyright (c) 2020 Archit Rungta
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import numpy as np
import math
from pytorch_neat.multi_env_eval import MultiEnvEvaluator


class StandardEnvEvaluator(MultiEnvEvaluator):
    def __init__(self, make_net, activate_net, max_rewards, batch_size=1, max_env_steps=None, make_env=None, envs=None):
        
        self.all_rewards = []
        self.max_rewards = max_rewards
        self.reward_idx = 0
        super().__init__(make_net, activate_net, batch_size=batch_size, max_env_steps=max_env_steps, make_env=make_env, envs=envs)

    def eval_genome(self, genome, config, debug=False):
        net = self.make_net(genome, config, self.batch_size)

        fitnesses = []
        for _ in range(self.batch_size):
            fitnesses.append([])

        states = [env.reset() for env in self.envs]
        dones = [False] * self.batch_size

        step_num = 0
        while True:
            step_num += 1
            if self.max_env_steps is not None and step_num == self.max_env_steps:
                break
            if debug:
                actions = self.activate_net(
                    net, states, debug=True, step_num=step_num)
            else:
                actions = self.activate_net(net, states)
            assert len(actions) == len(self.envs)
            for i, (env, action, done) in enumerate(zip(self.envs, actions, dones)):
                if not done:
                    state, reward, done, _ = env.step(action)
                    fitnesses[i].append(reward)
                    if not done:
                        states[i] = state
                    dones[i] = done
            if all(dones):
                break

        fitness = 0

        try:
            for fs in fitnesses:
                for fts in fs:
                    if (self.reward_idx < self.max_rewards):
                        self.all_rewards.append(fts)
                    else:
                        self.all_rewards[self.reward_idx % self.max_rewards] = fts
                    self.reward_idx += 1
        except MemoryError as error:
            # Output expected MemoryErrors.
            print(error)
        except Exception as exception:
            # Output unexpected Exceptions.
            print(exception, False)

        meanfitness = np.mean(self.all_rewards)
        stdfitness = np.std(self.all_rewards)

        for fs in fitnesses:
            lft = 0
            for i, fts in enumerate(fs):
                lft += (fts - meanfitness)/stdfitness
            fitness += lft
        
        # print (self.reward_idx)

        genome.val_fitness = super().eval_genome(genome, config, debug=debug)

        return fitness / len(fitnesses)