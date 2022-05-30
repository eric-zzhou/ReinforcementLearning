"""
Python code for the 2048 game that is specifically designed to work for AI and ML
Modified version of https://github.com/tjwei/2048-NN/blob/master/c2048.py
Tiny bit of help from https://github.com/lazuxd/playing-2048-with-minimax
"""
from random import random, randint, shuffle, choice
import numpy as np
import math
from copy import deepcopy

# a_moves = ["LEFT", "UP", "RIGHT", "DOWN"]
SNAKE_RATIO = 0.25
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


# #         empty, edge, smooth, matches,  mono, snake
# weights = [1, 0.75, 1, 0.75, 0.1, 2]

def flog2(val):
    return math.log2(val) if val != 0 else 0


class AdversarialGame:
    # ------------------------------------------------ Basic Game Stuff ------------------------------------------------
    # Game constructor with default 4x4 board
    def __init__(self, ws=[0.963, 0.5743, 0.1967, 0.2852, 0.4309, 0.4678], start_two=True,
                 matrix=np.zeros(shape=(4, 4), dtype='uint16')):
        self.grid = matrix
        self.score = 0
        self.end = False
        self.weights = ws
        # Initialize board with 2 random numbers
        if start_two:
            self.put_new_cell()
            self.put_new_cell()

    # Default equals function for objects
    def __eq__(self, other):
        for i in range(4):
            for j in range(4):
                if self.grid[i][j] != other.grid[i][j]:
                    return False
        return True

    # Set certain slot to a certain value
    def place_tile(self, r, c, val):
        self.grid[r][c] = val

    # Sets entire matrix
    def set_grid(self, grd):
        self.grid = deepcopy(grd)

    # Set weight
    def set_weights(self, ws):
        self.weights = ws

    # Returns matrix
    def get_grid(self):
        return deepcopy(self.grid)

    # Return coordinates of all empty spots
    def get_empty(self):
        spots = []
        for r in range(4):
            for c in range(4):
                if not self.grid[r, c]:
                    spots.append((r, c))
        return spots

    # Resets game
    def reset_game(self, start_two=True):
        self.grid = np.zeros(shape=(4, 4), dtype='uint16')
        self.score = 0
        self.end = False
        # Initialize board with 2 random numbers
        if start_two:
            self.put_new_cell()
            self.put_new_cell()

    # Print out board
    def display(self):
        s = "\n"
        wall = "+------" * 4 + "+"
        s += wall + "\n"
        for i in range(4):
            insides = "|".join("{:^6}".format(self.grid[i, j]) for j in range(4))
            s += f"|{insides}|\n{wall}\n"
        return s

    # Putting new cells on the board
    def put_new_cell(self):
        empty = self.get_empty()
        # If there are empty slots
        if empty:
            r, c = choice(empty)  # random location
            self.grid[r, c] = 2 if random() < 0.9 else 4  # 2 or 4 in the random location

        # Return number of empty slots
        return len(empty)

    # Checks if there are any possible moves remaining
    def any_possible_moves(self):
        # Looping through every spot
        for r in range(1, 4):
            for c in range(1, 4):
                current = self.grid[r, c]
                # Checks if there are any empty slots in self and 2 directions
                if not current:  # self
                    return True
                if c and current == self.grid[r, c - 1]:  # left
                    return True
                if r and current == self.grid[r - 1, c]:  # up
                    return True
        # If all the squares were checked and none were empty
        return False

    # Spawn new number on grid and returning whether the game ended
    def prepare_next_turn(self):
        empty_count = self.put_new_cell()
        return empty_count > 1 or self.any_possible_moves()

    # ------------------------------------------------ Game Moves Stuff ------------------------------------------------
    # Play a certain move and calculate score
    def move(self, direction):
        # Figure out which move to play
        if direction & 1:
            if direction & 2:
                cur_score = self.push_down()
            else:
                cur_score = self.push_up()
        else:
            if direction & 2:
                cur_score = self.push_right()
            else:
                cur_score = self.push_left()

        # Dealing with score
        if cur_score == -1:
            return 0, 0

        if cur_score > 0:
            self.score += cur_score

        # Check if game is over
        if not self.prepare_next_turn():
            self.end = True
        return 1, cur_score

    # Play a certain move without setting up for next move (for manual game)
    def game_move(self, direction):
        # Figure out which move to play
        if direction & 1:
            if direction & 2:
                cur_score = self.push_down()
            else:
                cur_score = self.push_up()
        else:
            if direction & 2:
                cur_score = self.push_right()
            else:
                cur_score = self.push_left()

        # Dealing with score
        if cur_score == -1:
            return 0

        if cur_score > 0:
            self.score += cur_score

        return 1

    # Push left command
    def push_left(self):
        # Variables
        moved, cur_score = False, 0

        # Looping through and moving each number accordingly
        for r in range(4):
            spot, prev = 0, 0
            for c in range(4):
                current = self.grid[r, c]
                if current:  # if there's something there
                    if current == prev:  # if the number is the same as the previous
                        self.grid[r, spot - 1] += current  # combine into one
                        cur_score += math.log2(current)  # update score
                        prev, moved = 0, True  # set variables
                    else:  # if the numbers are different
                        moved |= (spot != c)  # updates moved if unmoved
                        prev = self.grid[r, spot] = current  # update prev
                        spot += 1  # increment i

            # Fill the remaining right part with 0
            while spot < 4:
                self.grid[r, spot] = 0
                spot += 1

        # Return score or -1 if nothing moved
        return cur_score if moved else -1

    # Push right command
    def push_right(self):
        # Variables
        moved, cur_score = False, 0

        # Looping through and moving each number accordingly
        for r in range(4):
            spot, prev = 3, 0
            for c in range(3, -1, -1):
                current = self.grid[r, c]
                if current:  # if there's something there
                    if current == prev:  # if the number is the same as the previous
                        self.grid[r, spot + 1] += current  # combine into one
                        cur_score += math.log2(current)  # update score
                        prev, moved = 0, True  # set variables
                    else:  # if the numbers are different
                        moved |= (spot != c)  # updates moved if unmoved
                        prev = self.grid[r, spot] = current  # update prev
                        spot -= 1  # decrement i

            # Fill the remaining top part with
            while 0 <= spot:
                self.grid[r, spot] = 0
                spot -= 1

        # Return score or -1 if nothing moved
        return cur_score if moved else -1

    # Push up command
    def push_up(self):
        # Variables
        moved, cur_score = False, 0

        # Looping through and moving each number accordingly
        for c in range(4):
            spot, prev = 0, 0
            for r in range(4):
                current = self.grid[r, c]
                if current:  # if there's something there
                    if current == prev:  # if the number is the same as the previous
                        self.grid[spot - 1, c] += current  # combine into one
                        cur_score += math.log2(current)  # update score
                        prev, moved = 0, True  # set variables
                    else:  # if the numbers are different
                        moved |= (spot != r)  # updates moved if unmoved
                        prev = self.grid[spot, c] = current  # update prev
                        spot += 1  # increment i

            # Fill the remaining bottom part with 0
            while spot < 4:
                self.grid[spot, c] = 0
                spot += 1

        # Return score or -1 if nothing moved
        return cur_score if moved else -1

    # Push down command
    def push_down(self):
        # Variables
        moved, cur_score = False, 0

        # Looping through and moving each number accordingly
        for c in range(4):
            spot, prev = 3, 0
            for r in range(3, -1, -1):
                current = self.grid[r, c]
                if current:  # if there's something there
                    if current == prev:  # if the number is the same as the previous
                        self.grid[spot + 1, c] += current  # combine into one
                        cur_score += math.log2(current)  # update score
                        prev, moved = 0, True  # set variables
                    else:  # if the numbers are different
                        moved |= (spot != r)  # updates moved if unmoved
                        prev = self.grid[spot, c] = current  # update prev
                        spot -= 1  # decrement i

            # Fill the remaining top part with
            while 0 <= spot:
                self.grid[spot, c] = 0
                spot -= 1

        # Return score or -1 if nothing moved
        return cur_score if moved else -1

    # ------------------------------------------------- Fitness Stuff -------------------------------------------------
    # Follow snake pattern to help calculate fitness based on strategy
    def corner_traverse(self, corner):
        nums1 = nums2 = 0
        dec = (3, -1, -1)
        inc = (0, 4, 1)
        if corner >= 2:
            v_stuff = (dec, inc)
        else:
            v_stuff = (inc, dec)

        if corner % 2 == 0:
            h_stuff = (inc, dec)
        else:
            h_stuff = (dec, inc)

        switch = False
        multiplier = 1
        for j in range(v_stuff[0][0], v_stuff[0][1], v_stuff[0][2]):
            for i in range(h_stuff[switch][0], h_stuff[switch][1], h_stuff[switch][2]):
                cur = self.grid[j][i]
                nums1 += multiplier * cur
                multiplier *= SNAKE_RATIO

        switch = False
        multiplier = 1
        for i in range(h_stuff[0][0], h_stuff[0][1], h_stuff[0][2]):
            for j in range(v_stuff[switch][0], v_stuff[switch][1], v_stuff[switch][2]):
                cur = self.grid[j][i]
                nums2 += multiplier * cur
                multiplier *= SNAKE_RATIO
        # print(nums1, nums2)
        return max(nums1, nums2)

    # How different neighboring numbers are
    def grid_smoothness(self, corner):
        smoothness = 0
        empty_sqs = 0
        matches = 0
        mono = 0
        for i in range(4):
            for j in range(4):
                cur = self.grid[i][j]
                if cur:
                    rn = self.next_right(i, j)
                    if rn:
                        diff = math.log2(rn) - math.log2(cur)
                        if diff > 0:
                            smoothness -= abs(diff)
                            if (corner == 0) or (corner == 2):
                                mono -= math.log2(rn)
                        elif diff < 0:
                            smoothness -= abs(diff)
                            if (corner == 1) or (corner == 3):
                                mono -= math.log2(rn)
                        else:
                            matches += math.log2(cur)

                    dn = self.next_down(i, j)
                    if dn:
                        diff = math.log2(dn) - math.log2(cur)
                        if diff:
                            smoothness -= abs(diff)
                            if (corner == 0) or (corner == 1):
                                mono -= math.log2(dn)
                        elif diff < 0:
                            smoothness -= abs(diff)
                            if (corner == 2) or (corner == 3):
                                mono -= math.log2(dn)
                        else:
                            matches += math.log2(cur)
                else:
                    empty_sqs += 1
        return smoothness, empty_sqs, matches, mono

    # Traverse to next right tile that's not 0
    def next_right(self, r, c):
        while c < 3:
            rn = self.grid[r][c + 1]
            if rn == 0:
                c += 1
            else:
                return rn
        return 0

    # Traverse to next down tile that's not 0
    def next_down(self, r, c):
        while r < 3:
            dn = self.grid[r + 1][c]
            if dn == 0:
                r += 1
            else:
                return dn
        return 0

    # Adds log values of edge pieces
    def edges(self, corner):
        if corner == 0:
            return flog2(self.grid[0][0]) + flog2(self.grid[0][1]) + flog2(self.grid[0][2]) + flog2(self.grid[0][3]) \
                   + flog2(self.grid[1][0]) + flog2(self.grid[2][0]) + flog2(self.grid[3][0])
        elif corner == 1:
            return flog2(self.grid[0][0]) + flog2(self.grid[0][1]) + flog2(self.grid[0][2]) + flog2(self.grid[0][3]) \
                   + flog2(self.grid[1][3]) + flog2(self.grid[2][3]) + flog2(self.grid[3][3])
        elif corner == 2:
            return flog2(self.grid[0][0]) + flog2(self.grid[1][0]) + flog2(self.grid[2][0]) + flog2(self.grid[3][0]) \
                   + flog2(self.grid[3][1]) + flog2(self.grid[3][2]) + flog2(self.grid[3][3])
        else:
            return flog2(self.grid[0][3]) + flog2(self.grid[1][3]) + flog2(self.grid[2][3]) + flog2(self.grid[3][0]) \
                   + flog2(self.grid[3][1]) + flog2(self.grid[3][2]) + flog2(self.grid[3][3])

    # Find maximum number on the board
    def maximum(self):
        m = (0, -1, -1)
        for r in range(4):
            for c in range(4):
                cur = self.grid[r, c]
                if cur > m[0]:
                    m = (cur, r, c)
        return m

    def fitness(self):
        # count = 0
        # sum = 0
        # for i in range(4):
        #     for j in range(4):
        #         sum += self.grid[i][j]
        #         if self.grid[i][j]:
        #             count += 1
        # return int(sum / count)

        # Figure out which corner to use
        m, px, py = self.maximum()
        if self.grid[0][0] == m:
            corner = 0
        elif self.grid[0][3] == m:
            corner = 1
        elif self.grid[3][0] == m:
            corner = 2
        elif self.grid[3][3] == m:
            corner = 3
        else:
            corner = CORNER_MAPPING.get((px, py))

        # Get all the different parts
        snake = self.corner_traverse(corner)
        edge = self.edges(corner)
        smooth, empty, matches, mono = self.grid_smoothness(corner)

        return (snake * self.weights[5] + smooth * self.weights[2] + empty * self.weights[0] + matches * self.weights[3]
                + edge * self.weights[1] + mono * self.weights[4])


# Function to play random moves to test game code
def random_play(game, prt):
    moves = [0, 1, 2, 3]
    if prt:
        a_moves = ["LEFT", "UP", "RIGHT", "DOWN"]
        buff = [" ", "   ", " ", " "]
    while not game.end:
        shuffle(moves)
        for m in moves:
            if game.move(m):
                break
        if prt:
            print("              |")
            print("              v")
            print(buff[m] + "----------", a_moves[m], "----------", end="")
            print(game.display(), end="")
        print(game.score)
    return game.score


# Main function to test program
if __name__ == "__main__":
    g = AdversarialGame()
    score = random_play(g, True)
    print("Score: ", score)
