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

from common import *

@click.command()
@click.option("--n_generations", type=int, default=1000)
@click.option("--batch_size", type=int, default=1)
@click.option("--threads", type=int, default=1)
def run(n_generations, batch_size, threads):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    config_path = os.path.join(os.path.dirname(__file__), "neat.cfg")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    evaluator = StandardEnvEvaluator(
        make_net, activate_net, make_env=make_env, max_env_steps=max_env_steps
    )

    def eval_genomes(genomes, config):
        for _, genome in genomes:
            genome.fitness = evaluator.eval_genome(genome, config)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)
    logger = TensorBoardReporter("%s-standardise-%s-batch" % (env_name, str(batch_size)), "neat3.log", evaluator.eval_genome)
    pop.add_reporter(logger)

    peval = neat.ParallelEvaluator(threads, eval_genomes)

    pop.run(peval.eval_function, n_generations)



if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
