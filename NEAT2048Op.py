from game import OpGame
import neat
import os
import pickle
import math
import multiprocessing

CORNER_COORDS = [(0, 0), (0, 3), (3, 0), (3, 3)]
CORNER_DIST_THRESH = 2


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
                    moved, cur_score, combs = self.game.move(i)
                    if moved:
                        break

                m = self.game.maximum()

                snake = max(self.game.corner_traverse(0), self.game.corner_traverse(1), self.game.corner_traverse(2),
                            self.game.corner_traverse(3))

                fitness += snake

                if self.game.end:
                    self.game.display()
                    overall_fitness += fitness
                    print(f"{fitness}, {m}")
                    run = False
        return overall_fitness


def eval_genome(genome, conf):
    game = TwoGame()
    # print("___________________________________________________________________________________________________________")
    return game.train_ai(genome, conf)


def eval_genomes(genomes, conf):
    for i, genome_stuff in enumerate(genomes):
        genome_id, genome = genome_stuff
        print(f"\n\nindividual {i}: -----------------------------------------------------------------------------")
        genome.fitness = eval_genome(genome, conf)


def run_neat(conf):
    p = neat.Checkpointer.restore_checkpoint('snakeonly-250pop-459')
    # p = neat.Population(conf)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    p.add_reporter(neat.Checkpointer(generation_interval=5, time_interval_seconds=None,
                                     filename_prefix="snakeonly-250pop-"))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = p.run(pe.evaluate, 1)

    # winner = p.run(eval_genomes, 1000000000)

    with open("snakeonly_winner3.pickle", "wb") as f:
        pickle.dump(winner, f)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    run_neat(config)
