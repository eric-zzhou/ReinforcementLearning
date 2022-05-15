from game import ImprovedGame
import neat
import os
import pickle
import math
import multiprocessing


class TwoGame:
    def __init__(self):
        self.game = ImprovedGame()

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

                # Adjusting fitness after each step
                corner = -1
                order = self.game.ordered()

                # Making sure maximum is in one of the corners
                m, px, py = order[0]
                if self.game.grid[0][0] == m:
                    cur_score *= math.log2(m)
                    corner = 0
                elif self.game.grid[0][3] == m:
                    cur_score *= math.log2(m)
                    corner = 1
                elif self.game.grid[3][0] == m:
                    cur_score *= math.log2(m)
                    corner = 2
                elif self.game.grid[3][3] == m:
                    cur_score *= math.log2(m)
                    corner = 3

                # Snaking pattern
                if corner != -1:
                    smoothness = self.game.corner_traverse(corner)
                    if smoothness != 0:
                        cur_score *= smoothness

                fitness += cur_score

                if self.game.end:
                    self.game.display()
                    overall_fitness += fitness
                    print(f"{fitness}, {m}")
                    run = False
        return overall_fitness


# if self.game.end:
#     self.game.display()
#     fitness = self.game.score
#     maximum, count, order = self.game.ordered()
#
#     px = 0
#     py = 0
#     s = ""
#     for i in range(count):
#         m, px, py = order[0]
#         if m != maximum:
#             print(f"ERROR SOMETHING IS WRONG: {maximum} vs {m} ({px}, {py})")
#             exit()
#         if ((px == 0) or (px == 3)) and ((py == 0) or (px == 3)):
#             fitness *= math.log2(maximum)
#             s = f" * {math.log2(maximum)}"
#             # print(f"\tmultiplied: {px}, {py}, {(px == 0) or (px == 3)}, {(py == 0) or (py == 3)}, "
#             #       f"{((px == 0) or (px == 3)) and ((py == 0) or (py == 3))}")
#             order.pop(i)
#             break
#     if s == "":
#         order.pop(0)
#
#     smoothness = 0
#     right_spot = []
#     for num, x, y in order:
#         if abs(x - px) + abs(y - py) <= 1:
#             smoothness += math.log2(num)
#             right_spot.append(num)
#             px = x
#             py = y
#         else:
#             break
#     if smoothness != 0:
#         fitness *= smoothness
#
#     genome.fitness += fitness
#     print(f"{fitness}, {maximum}")
#     print(f"\tfitness: {self.game.score}{s}, {right_spot}")
#     run = False

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
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-6066')
    p = neat.Population(conf)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    p.add_reporter(neat.Checkpointer(125))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = p.run(pe.evaluate, 1000000000)

    # winner = p.run(eval_genomes, 1000000000)

    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    run_neat(config)
