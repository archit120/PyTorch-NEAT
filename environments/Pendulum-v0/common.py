# Copyright (c) 2018 Archit Rungta
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

import os

import click
import gym
import neat
import numpy as np
from pytorch_neat.discount_factor_eval import DiscountEnvEvaluator
from pytorch_neat.standardise_eval import StandardEnvEvaluator
from pytorch_neat.rewardtogo_eval import RewardToGoEnvEvaluator
from pytorch_neat.multi_env_eval import MultiEnvEvaluator
from pytorch_neat.neat_reporter import TensorBoardReporter
from pytorch_neat.recurrent_net import RecurrentNet

max_env_steps = 200

env_name = "Pendulum-v0"

def make_env():
    return gym.make(env_name)


def make_net(genome, config, bs):
    return RecurrentNet.create(genome, config, bs)


def activate_net(net, states):
    states = np.array(states)
    states[:, 2] /= 8
    # print(states)
    output = net.activate(states).numpy()
    res = (output-0.5)*4
    # print(res)
    return res

    # states = np.array(states)
    # # print(states.shape)

    # theta = np.arctan2(states[:, 1], states[:, 0])
    # states = np.concatenate(([theta], [states[:, 2]]), axis = 1)
    # # print(states.shape)
    # output = net.activate(states).numpy()
    # res = (output-0.5)*4
    # # print(res)
