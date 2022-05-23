from game import OpGame
import neat
import os
import pickle
from multiprocessing import Pool
import multiprocessing
import pandas
from pandas import DataFrame
import visualize

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
# count = 0
# values = []
# total = 0
# highest_score = 0

VALUES = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
initial_weights = [1, 1, 1, 0.1, 0.5, 0.1]
with open(r'C:\Users\ezhou\PycharmProjects\ReinforcedLearning\NEAT_2048\weights.pkl', 'rb') as f:
    weights = pickle.load(f)
WEIGHTS_COR = ["emptyw", "edgew", "smoothw", "matchw", "monow", "snakew"]


# todo https://github.com/xificurk/2048-ai, look in cpp file and search for heur_score
#  https://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048/22389702#22389702

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
                    self.game.display()
                    overall_fitness += fitness
                    # global count, total, values, highest_score
                    # if score > highest_score:
                    #     highest_score = score
                    # if count > 24499:
                    #     total += score
                    #     values.append(score)
                    # count += 1

                    print(f"{fitness}, {score}, {m}")
                    run = False
        return overall_fitness


def eval_genome(genome, conf):
    game = TwoGame()
    # print("___________________________________________________________________________________________________________")
    fitness = game.train_ai(genome, conf)
    # global count, values, total, highest_score
    # print(count, values, total, highest_score)
    return fitness


def eval_genomes(genomes, conf):
    for i, genome_stuff in enumerate(genomes):
        genome_id, genome = genome_stuff
        print(f"\n\nindividual {i}: -----------------------------------------------------------------------------")
        genome.fitness = eval_genome(genome, conf)


def run_neat(conf):
    for weight in WEIGHTS_COR[2:]:
        print(f"STARTING {weight}")
        # best_weight = 0
        # best_val = -1000000
        df = DataFrame()
        for val in VALUES:
            print(f"STARTING {val}")
            weights[WEIGHTS_COR.index(weight)] = val
            p = neat.Population(conf)
            p.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            p.add_reporter(stats)
            p.add_reporter(neat.Checkpointer(generation_interval=50, time_interval_seconds=None,
                                             filename_prefix=f"{weight}-{val}-"))

            pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
            winner = p.run(pe.evaluate, 50)

            # winner = p.run(eval_genomes, 50)
            # global count, values, total, highest_score
            # best = highest_score
            # round_stats = {
            #     "val": [val],
            #     "mean": [total / 500],
            #     "best": [highest_score]
            # }
            # for i in range(len(values)):
            #     round_stats[f"{i}"] = values[i]
            #
            # if best > best_val:
            #     best_val = best
            #     best_weight = val
            #
            # # Reset stuff
            # count = 0
            # values = []
            # total = 0
            # highest_score = 0
            # # print(mean, stdev, best)
            #
            # # with open(f"{weight}_{val}_winner.pickle", "wb") as f:
            # #     pickle.dump(winner, f)
            #
            # df2 = DataFrame(round_stats)
            # df = pandas.concat([df, df2], ignore_index=True, axis=0)
            try:
                visualize.draw_net(config=config, genome=winner, filename=f'{weight}-{val}-best.svg')
            except Exception as ex:
                print(ex)
                pass
            try:
                visualize.plot_stats(statistics=stats, filename=f'{weight}-{val}-fitnesses.svg')
            except Exception as ex:
                print(ex)
                pass
            try:
                visualize.plot_species(statistics=stats, filename=f'{weight}-{val}-species.svg')
            except Exception as ex:
                print(ex)
                pass
        # df.set_index("val", inplace=True)
        # df.to_excel(f'{weight}-checking.xlsx')
        ind = WEIGHTS_COR.index(weight)
        weights[ind] = initial_weights[ind]
        # print("Old weights:", weights)
        # weights[WEIGHTS_COR.index(weight)] = best_weight
        # pickle.dump(weights, open(r'C:\Users\ezhou\PycharmProjects\ReinforcedLearning\NEAT_2048\weights.pkl', 'wb'))
        # print("New weights:", weights)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    run_neat(config)
