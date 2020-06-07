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


class BaselineEnvEvaluator(MultiEnvEvaluator):
    def __init__(self, make_net, activate_net, baseline_net, gamma, batch_size=1, max_env_steps=None, make_env=None, envs=None):
        # baseline_net must by a pytorch model that takes state vector and outputs a number
        self.baseline_net = baseline_net

        super().__init__(make_net, activate_net, batch_size=batch_size, max_env_steps=max_env_steps, make_env=make_env, envs=envs)

    def eval_genome(self, genome, config, debug=False):
        net = self.make_net(genome[0], config, self.batch_size)

        fitnesses = np.zeros(self.batch_size)
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
                    fitnesses[i] += reward - self.e
                    if not done:
                        states[i] = state
                    dones[i] = done
            if all(dones):
                break

        genome.val_fitness = super().eval_genome(genome, config, debug=debug)

        return sum(fitnesses) / len(fitnesses)
