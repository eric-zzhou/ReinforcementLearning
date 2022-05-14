"""
Python code for the 2048 game that is specifically designed to work for AI and ML
models to be trained on
Modified version of https://github.com/Mekire/console-2048/blob/master/console2048.py
"""
import sys
from random import random, randint, shuffle
import numpy as np
import math


# Game object with 2048 game
class Game:
    # Game constructor with default 4x4 board
    def __init__(self, cols=4, rows=4, start_two=True):
        self.grid = np.zeros(shape=(rows, cols), dtype='uint16')
        self.score = 0
        self.end = False
        # Initialize board with 2 random numbers
        if start_two:
            self.put_new_cell()
            self.put_new_cell()

    # Flattens board to input into nn
    def flatten(self):
        flat = []
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                flat.append(self.grid[i][j])
        return flat

    # Find maximum number on the board
    def max(self):
        m = 0
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i, j] > m:
                    m = self.grid[i, j]
        return m

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
            return 0

        if cur_score > 0:
            self.score += cur_score

        # Check if game is over
        if not self.prepare_next_turn():
            self.end = True
        return 1

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

    # Set certain slot to a certain value
    def set_grid(self, grd):
        self.grid = grd

    # Print out board
    def display(self):
        print("")
        wall = "+------" * self.grid.shape[1] + "+"
        print(wall)
        for i in range(self.grid.shape[0]):
            insides = "|".join("{:^6}".format(self.grid[i, j]) for j in range(self.grid.shape[1]))
            print(f"|{insides}|")
            print(wall)

    # Push left command
    def push_left(self):
        # Variables
        moved, cur_score = False, 0
        rows, columns = self.grid.shape[0], self.grid.shape[1]

        # Looping through and moving each number accordingly
        for r in range(rows):
            spot, prev = 0, 0
            for c in range(columns):
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
            while spot < columns:
                self.grid[r, spot] = 0
                spot += 1
        # Return score or -1 if nothing moved
        return cur_score if moved else -1

    # Push right command
    def push_right(self):
        # Variables
        moved, cur_score = False, 0
        rows, columns = self.grid.shape[0], self.grid.shape[1]

        # Looping through and moving each number accordingly
        for r in range(rows):
            spot, prev = columns - 1, 0
            for c in range(columns - 1, -1, -1):
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
            # Fill the remaining left part with 0
            while 0 <= spot:
                self.grid[r, spot] = 0
                spot -= 1
        # Return score or -1 if nothing moved
        return cur_score if moved else -1

    # Push up command
    def push_up(self):
        # Variables
        moved, cur_score = False, 0
        rows, columns = self.grid.shape[0], self.grid.shape[1]

        # Looping through and moving each number accordingly
        for c in range(columns):
            spot, prev = 0, 0
            for r in range(rows):
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
            while spot < rows:
                self.grid[spot, c] = 0
                spot += 1
        # Return score or -1 if nothing moved
        return cur_score if moved else -1

    # Push down command
    def push_down(self):
        # Variables
        moved, cur_score = False, 0
        rows, columns = self.grid.shape[0], self.grid.shape[1]

        # Looping through and moving each number accordingly
        for c in range(columns):
            spot, prev = rows - 1, 0
            for r in range(rows - 1, -1, -1):
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
            # Fill the remaining top part with 0
            while 0 <= spot:
                self.grid[spot, c] = 0
                spot -= 1
        # Return score or -1 if nothing moved
        return cur_score if moved else -1

    # Putting new cells on the board
    def put_new_cell(self):
        i_s = []
        j_s = []

        # Find all empty slots
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if not self.grid[i, j]:
                    i_s.append(i)
                    j_s.append(j)
        # print(i_s)
        # print(j_s)
        # print_grid(grid)
        # If there are empty slots
        if i_s:
            r = randint(0, len(i_s) - 1)  # random location
            self.grid[i_s[r], j_s[r]] = 2 if random() < 0.9 else 4  # 2 or 4 in the random location

        # Return number of empty slots
        return len(i_s)

    # Checks if there are any possible moves remaining
    def any_possible_moves(self):
        rows, columns = self.grid.shape[0], self.grid.shape[1]
        # Looping through every spot
        for r in range(1, rows):
            for c in range(1, columns):
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


# Improved game
class ImprovedGame:
    # Game constructor with default 4x4 board
    def __init__(self, cols=4, rows=4, start_two=True):
        self.grid = np.zeros(shape=(rows, cols), dtype='uint16')
        self.score = 0
        self.end = False
        # Initialize board with 2 random numbers
        if start_two:
            self.put_new_cell()
            self.put_new_cell()

    def reset_game(self, cols=4, rows=4, start_two=True):
        self.grid = np.zeros(shape=(rows, cols), dtype='uint16')
        self.score = 0
        self.end = False
        # Initialize board with 2 random numbers
        if start_two:
            self.put_new_cell()
            self.put_new_cell()

    # Traverse from corner
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
        cont = True
        prev = 1000000000
        for j in range(v_stuff[0][0], v_stuff[0][1], v_stuff[0][2]):
            for i in range(h_stuff[switch][0], h_stuff[switch][1], h_stuff[switch][2]):
                cur = self.grid[j][i]
                if cur < prev:  # todo change condition here for continuing
                    nums1 += math.log2(cur)
                    prev = cur
                else:
                    cont = False
                    break
            if cont:
                switch = not switch
            else:
                break

        switch = False
        cont = True
        prev = 1000000000
        for i in range(h_stuff[0][0], h_stuff[0][1], h_stuff[0][2]):
            for j in range(v_stuff[switch][0], v_stuff[switch][1], v_stuff[switch][2]):
                cur = self.grid[j][i]
                if cur < prev:
                    nums2 += math.log2(cur)
                    prev = cur
                else:
                    cont = False
                    break
            if cont:
                switch = not switch
            else:
                break

        return max(nums1, nums2)

    # Flattens board to input into nn
    def flatten(self):
        flat = []
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                curr = self.grid[i][j]
                if (curr != 0):
                    flat.append(math.log2(curr))
                else:
                    flat.append(0)
        logmax = max(flat)
        for i in range(len(flat)):
            flat[i] = flat[i] / logmax
        return flat

    # Find maximum number on the board
    def ordered(self):
        m = []
        max = 0
        count = 0
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                cur = self.grid[i, j]
                if cur != 0:
                    m.append((cur, i, j))
                if cur > max:
                    max = cur
                    count = 1
                elif cur == max:
                    count += 1
        m.sort(reverse=True)
        return max, count, m

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

    # Set certain slot to a certain value
    def set_grid(self, grd):
        self.grid = grd

    # Print out board
    def display(self):
        print("")
        wall = "+------" * self.grid.shape[1] + "+"
        print(wall)
        for i in range(self.grid.shape[0]):
            insides = "|".join("{:^6}".format(self.grid[i, j]) for j in range(self.grid.shape[1]))
            print(f"|{insides}|")
            print(wall)

    # Push left command
    def push_left(self):
        # Variables
        moved, cur_score = False, 0
        rows, columns = self.grid.shape[0], self.grid.shape[1]

        # Looping through and moving each number accordingly
        for r in range(rows):
            spot, prev = 0, 0
            for c in range(columns):
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
            while spot < columns:
                self.grid[r, spot] = 0
                spot += 1
        # Return score or -1 if nothing moved
        return cur_score if moved else -1

    # Push right command
    def push_right(self):
        # Variables
        moved, cur_score = False, 0
        rows, columns = self.grid.shape[0], self.grid.shape[1]

        # Looping through and moving each number accordingly
        for r in range(rows):
            spot, prev = columns - 1, 0
            for c in range(columns - 1, -1, -1):
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
            # Fill the remaining left part with 0
            while 0 <= spot:
                self.grid[r, spot] = 0
                spot -= 1
        # Return score or -1 if nothing moved
        return cur_score if moved else -1

    # Push up command
    def push_up(self):
        # Variables
        moved, cur_score = False, 0
        rows, columns = self.grid.shape[0], self.grid.shape[1]

        # Looping through and moving each number accordingly
        for c in range(columns):
            spot, prev = 0, 0
            for r in range(rows):
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
            while spot < rows:
                self.grid[spot, c] = 0
                spot += 1
        # Return score or -1 if nothing moved
        return cur_score if moved else -1

    # Push down command
    def push_down(self):
        # Variables
        moved, cur_score = False, 0
        rows, columns = self.grid.shape[0], self.grid.shape[1]

        # Looping through and moving each number accordingly
        for c in range(columns):
            spot, prev = rows - 1, 0
            for r in range(rows - 1, -1, -1):
                current = self.grid[r, c]
                if current:  # if there's something there
                    if current == prev:  # if the number is the same as the previous
                        self.grid[spot + 1, c] += current  # combine into one
                        cur_score += math.log2(current)  # update score
                        prev, moved = 0, True  # set variables
                    else:  # if the numbers are different
                        moved |= (spot != r)  # updates moved if unmoved
                        # todo: if moved, then add to list to check for same number later
                        prev = self.grid[spot, c] = current  # update prev
                        spot -= 1  # decrement i
            # Fill the remaining top part with
            while 0 <= spot:
                self.grid[spot, c] = 0
                spot -= 1
        # Return score or -1 if nothing moved
        return cur_score if moved else -1

    # Putting new cells on the board
    def put_new_cell(self):
        i_s = []
        j_s = []

        # Find all empty slots
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if not self.grid[i, j]:
                    i_s.append(i)
                    j_s.append(j)
        # print(i_s)
        # print(j_s)
        # print_grid(grid)
        # If there are empty slots
        if i_s:
            r = randint(0, len(i_s) - 1)  # random location
            self.grid[i_s[r], j_s[r]] = 2 if random() < 0.9 else 4  # 2 or 4 in the random location

        # Return number of empty slots
        return len(i_s)

    # Checks if there are any possible moves remaining
    def any_possible_moves(self):
        rows, columns = self.grid.shape[0], self.grid.shape[1]
        # Looping through every spot
        for r in range(1, rows):
            for c in range(1, columns):
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
            game.display()
        print(game.score)
    return game.score


# Main function to test program
if __name__ == "__main__":
    g = Game()
    score = random_play(g, True)
    print("Score: ", score)
