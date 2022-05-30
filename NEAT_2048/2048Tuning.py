import logging
import time

from game import OpGame
import neat
import os
import pickle
import multiprocessing
import winsound
import pandas
from pandas import DataFrame
import visualize
import logging

logging.basicConfig(level=logging.INFO)

CORNER_MAPPING = {
    (0, 1): 0,
    (1, 0): 0,
    (1, 1): 0,
    (0, 2): 1,
    (1, 2): 1,
    (1, 3): 1,
    (2, 0): 2,
    (2, 1): 2,
    (3, 1): 2,
    (2, 2): 3,
    (2, 3): 3,
    (3, 2): 3
}

# move these to a pickle file (count, values, total, high_score)
# pickle.dump((0, [], 0, 0), open(r'C:\Users\ezhou\PycharmProjects\ReinforcedLearning\NEAT_2048\values.pkl', 'wb'))
values = []
total = 0
highest_score = 0

VALUES = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
with open(r'/NEAT_2048/NEAT_2048/weights.pkl', 'rb') as f:
    initial_weights = weights = pickle.load(f)
WEIGHTS_COR = ["emptyw", "edgew", "smoothw", "matchw", "monow", "snakew"]


class TwoGame:
    def __init__(self):
        self.game = OpGame()

    def train_ai(self, genome, conf):
        net = neat.nn.FeedForwardNetwork.create(genome, conf)
        overall_fitness = 0

        for games in range(5):
            self.game.reset_game()
            run = True
            fitness = 0
            while run:
                # Using neural network to make a move
                output = net.activate(tuple(self.game.flatten()))
                output_tuples = []
                for i, o in enumerate(output):
                    output_tuples.append((o, i))
                output_tuples.sort(reverse=True)
                for o, i in output_tuples:
                    moved, cur_score = self.game.move(i)
                    if moved:
                        break

                # Get order of squares
                order = self.game.ordered()

                # Figure out which corner to use
                m, px, py = order[0]
                if self.game.grid[0][0] == m:
                    corner = 0
                elif self.game.grid[0][3] == m:
                    corner = 1
                elif self.game.grid[3][0] == m:
                    corner = 2
                elif self.game.grid[3][3] == m:
                    corner = 3
                else:
                    corner = CORNER_MAPPING.get((px, py))
                snake = self.game.corner_traverse(corner)
                edge = self.game.edges(corner)
                smooth, empty, matches, mono = self.game.grid_smoothness(corner)

                fitness += (snake * weights[5] + smooth * weights[2] + empty * weights[0] + matches * weights[3] + edge
                            * weights[1] + mono * weights[4])

                if self.game.end:
                    score = self.game.score
                    logging.info(self.game.display() + f"{fitness}, {score}, {m}\n")
                    overall_fitness += fitness
                    global total, values, highest_score
                    if score > highest_score:
                        highest_score = score
                    total += score
                    values.append(score)
                    run = False
        return overall_fitness


def eval_genome(genome, conf):
    game = TwoGame()
    # print("___________________________________________________________________________________________________________")
    fitness = game.train_ai(genome, conf)
    return fitness


def eval_genomes(genomes, conf):
    for i, genome_stuff in enumerate(genomes):
        genome_id, genome = genome_stuff
        logging.info(
            f"\n\nindividual {i}: -----------------------------------------------------------------------------")
        genome.fitness = eval_genome(genome, conf)


def run_neat(conf):
    for weight in WEIGHTS_COR[2:3]:
        logging.info(f"STARTING {weight}")
        df = DataFrame()
        for val in VALUES:
            logging.info(f"STARTING {val}")
            # weights[WEIGHTS_COR.index(weight)] = val
            # p = neat.Population(conf)

            p = neat.Checkpointer.restore_checkpoint(
                f'/NEAT_2048/NEAT_2048/tuning\\{weight}-{val}-49')
            p.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            p.add_reporter(stats)
            # p.add_reporter(neat.Checkpointer(generation_interval=50, time_interval_seconds=None,
            #                                  filename_prefix=f"tuning\\{weight}-{val}-"))

            # pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
            # winner = p.run(pe.evaluate, 50)

            winner = p.run(eval_genomes, 1)
            global values, total, highest_score
            # best = highest_score
            round_stats = {
                "val": [val],
                "mean": [total / len(values)],
                "best": [highest_score]
            }
            for i in range(len(values)):
                round_stats[f"{i}"] = values[i]

            # Reset stuff
            values = []
            total = 0
            highest_score = 0
            # print(mean, stdev, best)

            df2 = DataFrame(round_stats)
            df = pandas.concat([df, df2], ignore_index=True, axis=0)
            # try:
            #     visualize.draw_net(config=config, genome=winner, filename=f'tuning\\{weight}-{val}-best.svg')
            # except Exception as ex:
            #     print(ex)
            #     pass
            # try:
            #     visualize.plot_stats(statistics=stats, filename=f'tuning\\{weight}-{val}-fitnesses.svg')
            # except Exception as ex:
            #     print(ex)
            #     pass
            # try:
            #     visualize.plot_species(statistics=stats, filename=f'tuning\\{weight}-{val}-species.svg')
            # except Exception as ex:
            #     print(ex)
            #     pass
        df.set_index("val", inplace=True)
        df.to_excel(f'tuning\\{weight}-checking.xlsx')
        # ind = WEIGHTS_COR.index(weight)
        # weights[ind] = initial_weights[ind]


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    run_neat(config)
    while True:
        winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
        time.sleep(0.025)
