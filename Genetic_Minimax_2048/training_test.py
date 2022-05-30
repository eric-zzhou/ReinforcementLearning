import pickle
import numpy as np
from game import AdversarialGame
from Minimax2048 import get_best_move
import logging
import winsound
import time

logging.basicConfig(level=logging.INFO)

start_time = time.time()

# Make game
a_moves = ["LEFT", "UP", "RIGHT", "DOWN"]

# Initializing local game
g = AdversarialGame(start_two=True)
for i in range(5):
    fitness = 0
    g.reset_game()
    while True:
        output = get_best_move(g, 5)
        g.move(output)
        # logging.info(f"Chosen move: {a_moves[output]}")

        if g.end:
            fitness += g.score
            logging.info(g.display())
            logging.info(g.score)
            break

while True:
    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
    time.sleep(0.025)
