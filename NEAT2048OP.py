from game import OpGame
import neat
import os
import pickle
import multiprocessing
import visualize
import logging
import winsound
import time

logging.basicConfig(level=logging.WARNING)
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
NUM_GAME = 5
#         empty, edge, smooth, matches,  mono, snake
weights = [1, 0.75, 1, 0.75, 0.1, 2]


def save_stats(stats):
    try:
        visualize.plot_stats(statistics=stats, filename='opfitnesses.svg')
    except Exception as ex:
        logging.warning(ex)
        pass
    try:
        visualize.plot_species(statistics=stats, filename='opspecies.svg')
    except Exception as ex:
        logging.warning(ex)
        pass


class TwoGame:
    def __init__(self):
        self.game = OpGame()

    def train_ai(self, genome, conf):
        net = neat.nn.FeedForwardNetwork.create(genome, conf)
        overall_fitness = 0

        for games in range(NUM_GAME):
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
                    logging.info(self.game.display())
                    overall_fitness += fitness
                    logging.info(f"{fitness}, {score}, {m}")
                    run = False
        # return sum(overall_fitness)
        return overall_fitness / NUM_GAME


def eval_genome(genome, conf):
    try:
        game = TwoGame()
        fitness = game.train_ai(genome, conf)
        if fitness > 300000:
            with open(f"op-{fitness}.pickle", "wb") as f:
                pickle.dump(genome, f)
        return fitness
    except:
        return 0


def eval_genomes(genomes, conf):
    for i, genome_stuff in enumerate(genomes):
        genome_id, genome = genome_stuff
        logging.debug(f"\n\nindividual {i}: --------------------------------------------------------------------------")
        genome.fitness = eval_genome(genome, conf)


def run_neat(conf):
    try:
        p = neat.Checkpointer.restore_checkpoint('op-100pop-3259')
        # p = neat.Population(conf)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(generation_interval=5, time_interval_seconds=None,
                                         filename_prefix=f"op-100pop-"))
        pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
        winner = p.run(pe.evaluate, 100000000000000000)

        # winner = p.run(eval_genomes, 100000000000000)
    except:
        save_stats(stats)
        exit()
    with open("op_winner.pickle", "wb") as f:
        pickle.dump(winner, f)

    # try:
    #     visualize.draw_net(config=config, genome=winner, filename='best.svg')
    # except Exception as ex:
    #     logging.warning(ex)
    #     pass
    save_stats(stats)


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
