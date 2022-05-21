from game import Game
import neat
import os
import pickle


class TwoGame:
    def __init__(self):
        self.game = Game()

    def train_ai(self, genome, conf):
        net = neat.nn.FeedForwardNetwork.create(genome, conf)

        run = True
        while run:
            output = net.activate(tuple(self.game.flatten()))
            output_tuples = []
            for i, o in enumerate(output):
                output_tuples.append((o, i))
            output_tuples.sort(reverse=True)
            for o, i in output_tuples:
                if self.game.move(i):
                    # a_moves = ["LEFT", "UP", "RIGHT", "DOWN"]
                    # print(a_moves[i])
                    break
            # self.game.display()
            # print(self.game.end)

            if self.game.end:
                genome.fitness += self.game.score
                print(f"{int(self.game.score)}, {self.game.max()}")
                if int(self.game.max()) >= 2048:
                    with open("", "w") as f:
                        f.write(f"Got 2048: {int(self.game.score)}, {self.game.max()}")
                run = False


def eval_genomes(genomes, conf):
    for genome_id, genome in genomes:
        genome.fitness = 0
        game = TwoGame()
        game.train_ai(genome, conf)


def run_neat(conf):
    p = neat.Checkpointer.restore_checkpoint('checkpoints/neat-checkpoint-12043-250pop')
    # p = neat.Population(conf)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    p.add_reporter(neat.Checkpointer(125))

    winner = p.run(eval_genomes, 1)
    with open("checkpoints/best.pickle", "wb") as f:
        pickle.dump(winner, f)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    run_neat(config)

