"""
Genetic algorithm trying to tune values

Parallel evaluator is slightly modified version of NEAT's
"""

import random
from game import AdversarialGame
from Minimax2048 import get_best_move
import pickle
import multiprocessing
import logging
import time
import sys
import os
import winsound

# Logging stuff
formatter = logging.Formatter(fmt='%(asctime)s,%(msecs)d %(levelname)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("TrainingFileLog")
file_handler = logging.FileHandler(filename='GA_Tuning_Log.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.WARNING)
logger2 = logging.getLogger("TrainingConsoleLog")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger2.addHandler(stream_handler)
logger2.setLevel(logging.INFO)

# Constants
POP_SIZE = 50  # population size
MAX_GEN = 100000000000000000  # max number of generations to run through
TOP_KEEP = 0.1  # amount of the best to be kept for next gen
TOP_BREED = 0.2  # amount of the best to be used for breeding
INTRO_RANDOM = 0.1  # amount of the best to be used for breeding
GOAL_FITNESS = 10000000000  # goal for fitness of function
MUTATION_VARIATION = 0.25  # amount to mutate by
VARS = 6  # total number of variables in the function
CHECKPOINT_NUM = 21  # number for checkpoint, 0 if none

# Get bounds for mutation
mut_s = 1 - MUTATION_VARIATION
mut_l = 1 + MUTATION_VARIATION


# ranked_sol = []


class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function, timeout=None):
        """
        eval_function should take one argument, a tuple of
        (genome object, config object), and return
        a single float (the genome's fitness).
        """
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.timeout = timeout
        self.pool = multiprocessing.Pool(num_workers)

    def __del__(self):
        self.pool.close()
        self.pool.join()

    def evaluate(self, weights):
        jobs = []
        sols = []
        avg = 0
        for weight in weights:
            jobs.append(self.pool.apply_async(self.eval_function, [weight]))

        # assign the fitness back to each genome
        for job, weight in zip(jobs, weights):
            fit = job.get(timeout=self.timeout)
            sols.append((fit, weight))
            avg += fit / POP_SIZE
        sols.sort(reverse=True)
        return avg, sols


# Determining fitness of solution
def fitness(weights):
    try:
        fit = 0
        g = AdversarialGame(ws=weights)
        for i in range(2):
            g.reset_game()
            while True:
                output = get_best_move(g, 3)
                g.move(output)
                logger.debug(g.display())
                logger2.debug(g.display())
                if g.end:
                    fit += g.score
                    logger.info(g.display())
                    logger2.info(g.display())
                    logger.info(f"score: {g.score}, weights: {weights}")
                    logger2.info(f"score: {g.score}, weights: {weights}")
                    break
        return fit
    except:
        return 0


if __name__ == "__main__":
    try:
        if CHECKPOINT_NUM:
            start_gen = CHECKPOINT_NUM
            with open(f'cp-{CHECKPOINT_NUM}.pickle', 'rb') as f:
                sol = pickle.load(f)
        else:
            # Generate random starting solutions
            start_gen = 1
            sol = []
            for i in range(POP_SIZE):
                sol.append([random.randint(0, 10000),
                            random.randint(0, 10000),
                            random.randint(0, 10000),
                            random.randint(0, 10000),
                            random.randint(0, 10000),
                            random.randint(0, 10000)])

        # Running through generations
        start_time = time.time()
        pe = ParallelEvaluator(multiprocessing.cpu_count() - 1, fitness)
        for gen in range(start_gen, MAX_GEN + 1):
            # Get fitness and rank solutions (best to worst)
            avg, ranked_sol = pe.evaluate(sol)
            # for s in sol:
            #     ranked_sol.append((fitness(s), s))
            # ranked_sol.sort(reverse=True)

            # Print to see the best solution and fitness for each gen
            logger.warning(
                f"------------------------- Generation {gen}, Time: {time.time() - start_time}, "
                f"Mean fitness: {avg} -------------------------")
            logger2.warning(
                f"------------------------- Generation {gen}, Time: {time.time() - start_time}, "
                f"Mean fitness: {avg} -------------------------")
            logger.warning(f"ranked solutions: {ranked_sol}")
            logger2.warning(f"ranked solutions: {ranked_sol}")
            start_time = time.time()

            # # Stop if it is past the goal
            # if ranked_sol[0][0] > GOAL_FITNESS:
            #     break

            # New population for next generation
            kept = int(POP_SIZE * TOP_KEEP)  # amount to keep
            # keeping some of the best from the gen
            sol = []
            for i in range(kept):
                sol.append(ranked_sol[i][1])

            # Introduce some randomness to prevent it from all being the same thing
            kept += int(POP_SIZE * INTRO_RANDOM)
            for i in range(int(kept / 2)):
                sol.append([random.randint(0, 10000),
                            random.randint(0, 10000),
                            random.randint(0, 10000),
                            random.randint(0, 10000),
                            random.randint(0, 10000),
                            random.randint(0, 10000)])

            # Make matrix to track vars separately in best solutions
            inds = []
            for v in range(VARS):
                inds.append([])
            for i in range(int(POP_SIZE * TOP_BREED)):
                inds[0].append(ranked_sol[i][1][0])
                inds[1].append(ranked_sol[i][1][1])
                inds[2].append(ranked_sol[i][1][2])
                inds[3].append(ranked_sol[i][1][3])
                inds[4].append(ranked_sol[i][1][4])
                inds[5].append(ranked_sol[i][1][5])

            # Creating new population with crossover and mutation
            for i in range(kept, POP_SIZE):
                sol.append([int(random.choice(inds[0]) * random.uniform(mut_s, mut_l)),
                            int(random.choice(inds[1]) * random.uniform(mut_s, mut_l)),
                            int(random.choice(inds[2]) * random.uniform(mut_s, mut_l)),
                            int(random.choice(inds[3]) * random.uniform(mut_s, mut_l)),
                            int(random.choice(inds[4]) * random.uniform(mut_s, mut_l)),
                            int(random.choice(inds[5]) * random.uniform(mut_s, mut_l))])

            if gen % 1 == 0:
                with open(f"cp-{gen + 1}.pickle", "wb") as f:
                    pickle.dump(sol, f)
                logger.info(f"Checkpoint {gen + 1} was added!")
                logger2.info(f"Checkpoint {gen + 1} was added!")
    except Exception as e:
        logger.error(e)
        logger2.error(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error(exc_type, fname, exc_tb.tb_lineno)
        logger2.error(exc_type, fname, exc_tb.tb_lineno)
    finally:
        while True:
            winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
            time.sleep(0.025)
