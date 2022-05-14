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

        for games in range(5):
            self.game.reset_game()
            run = True
            while run:
                output = net.activate(tuple(self.game.flatten()))
                output_tuples = []
                for i, o in enumerate(output):
                    output_tuples.append((o, i))
                output_tuples.sort(reverse=True)
                for o, i in output_tuples:
                    moved, cur_score = self.game.move(i)
                    if moved:
                        # a_moves = ["LEFT", "UP", "RIGHT", "DOWN"]
                        # print(a_moves[i])
                        break
                # self.game.display()
                # print(self.game.end)
                # todo adjust fitness after every step
                fitness = 0
                removed = False
                fitness += cur_score
                maximum, count, order = self.game.ordered()

                # Making sure maximum is in one of the corners
                for i in range(count):
                    m, px, py = order[0]
                    if m != maximum:
                        print(f"ERROR SOMETHING IS WRONG: {maximum} vs {m} ({px}, {py})")
                        exit()
                    if ((px == 0) or (px == 3)) and ((py == 0) or (px == 3)):
                        fitness *= math.log2(maximum)
                        order.pop(i)
                        removed = True
                        break
                if removed:  # todo finish the snaking part
                    smoothness = 0
                    if smoothness != 0:
                        fitness *= smoothness

                if self.game.end:
                    self.game.display()
                    genome.fitness += fitness
                    print(f"{fitness}, {maximum}")
                    run = False


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

def eval_genomes(genomes, conf):
    for i, genome_stuff in enumerate(genomes):
        genome_id, genome = genome_stuff
        genome.fitness = 0
        game = TwoGame()
        print(f"\n\nindividual {i}: -----------------------------------------------------------------------------")
        game.train_ai(genome, conf)


def run_neat(conf):
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-6066')
    p = neat.Population(conf)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    p.add_reporter(neat.Checkpointer(125))

    # todo multiprocessing might have problems, eval_genomes vs eval_genome
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genomes)

    winner = p.run(pe.eval_genomes, 1000000)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    run_neat(config)
