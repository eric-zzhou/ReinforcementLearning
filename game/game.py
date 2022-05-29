"""
Python code for the 2048 game that is specifically designed to work for AI and ML
models to be trained on
Modified version of https://github.com/tjwei/2048-NN/blob/master/c2048.py
"""
from random import random, randint, shuffle
import numpy as np
import math

# todo tune match fitness constant
MATCH_FIT_CONST = 0.3
SNAKE_RATIO = 0.25


def flog2(val):
    return math.log2(val) if val != 0 else 0


# Game object with 2048 game
class Game:
    # Game constructor with default 4x4 board
    def __init__(self, start_two=True):
        self.grid = np.zeros(shape=(4, 4), dtype='uint16')
        self.score = 0
        self.end = False
        # Initialize board with 2 random numbers
        if start_two:
            self.put_new_cell()
            self.put_new_cell()

    # Flattens board to input into nn
    def flatten(self):
        flat = []
        for i in range(4):
            for j in range(4):
                flat.append(self.grid[i][j])
        return flat

    # Find maximum number on the board
    def max(self):
        m = 0
        for i in range(4):
            for j in range(4):
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
        wall = "+------" * 4 + "+"
        print(wall)
        for i in range(4):
            insides = "|".join("{:^6}".format(self.grid[i, j]) for j in range(4))
            print(f"|{insides}|")
            print(wall)

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
        for i in range(4):
            for j in range(4):
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


# Improved game
class ImprovedGame:
    # Game constructor with default 4x4 board
    def __init__(self, start_two=True):
        self.grid = np.zeros(shape=(4, 4), dtype='uint16')
        self.score = 0
        self.end = False
        # Initialize board with 2 random numbers
        if start_two:
            self.put_new_cell()
            self.put_new_cell()

    def reset_game(self, start_two=True):
        self.grid = np.zeros(shape=(4, 4), dtype='uint16')
        self.score = 0
        self.end = False
        # Initialize board with 2 random numbers
        if start_two:
            self.put_new_cell()
            self.put_new_cell()

    # Flattens board to input into nn
    def flatten(self):
        flat = []
        for i in range(4):
            for j in range(4):
                curr = self.grid[i][j]
                if curr != 0:
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
        for i in range(4):
            for j in range(4):
                cur = self.grid[i, j]
                if cur != 0:
                    m.append((cur, i, j))
        m.sort(reverse=True)
        return m

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

        vals = self.ordered()

        switch = False
        cont = True
        counter = 0
        for j in range(v_stuff[0][0], v_stuff[0][1], v_stuff[0][2]):
            for i in range(h_stuff[switch][0], h_stuff[switch][1], h_stuff[switch][2]):
                cur = self.grid[j][i]
                if counter >= len(vals):
                    cont = False
                    break
                if cur == vals[counter][0]:
                    # todo change condition here for snaking fitness, maybe do the factor thing
                    #  where if the next one is not the exact next largest, it adds a fraction but keeps going
                    nums1 += math.log2(cur)
                    counter += 1
                else:
                    cont = False
                    break
            if cont:
                switch = not switch
            else:
                break

        switch = False
        cont = True
        counter = 0
        for i in range(h_stuff[0][0], h_stuff[0][1], h_stuff[0][2]):
            for j in range(v_stuff[switch][0], v_stuff[switch][1], v_stuff[switch][2]):
                cur = self.grid[j][i]
                if counter >= len(vals):
                    cont = False
                    break
                if cur == vals[counter][0]:
                    nums2 += math.log2(cur)
                    counter += 1
                else:
                    cont = False
                    break
            if cont:
                switch = not switch
            else:
                break
        # print(nums1, nums2)
        return max(nums1, nums2)

    # Play a certain move and calculate score
    def move(self, direction):
        # Figure out which move to play
        if direction & 1:
            if direction & 2:
                cur_score, combs = self.push_down()
            else:
                cur_score, combs = self.push_up()
        else:
            if direction & 2:
                cur_score, combs = self.push_right()
            else:
                cur_score, combs = self.push_left()

        # Dealing with score
        if cur_score == -1:
            return 0, 0, combs

        if cur_score > 0:
            self.score += cur_score

        # Check if game is over
        if not self.prepare_next_turn():
            self.end = True
        return 1, cur_score, combs

    # Play a certain move without setting up for next move (for manual game)
    def game_move(self, direction):
        # Figure out which move to play
        if direction & 1:
            if direction & 2:
                cur_score, _ = self.push_down()
            else:
                cur_score, _ = self.push_up()
        else:
            if direction & 2:
                cur_score, _ = self.push_right()
            else:
                cur_score, _ = self.push_left()

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
        s = "\n"
        wall = "+------" * 4 + "+"
        s += wall + "\n"
        for i in range(4):
            insides = "|".join("{:^6}".format(self.grid[i, j]) for j in range(4))
            s += f"|{insides}|\n{wall}\n"
        print(s, end="")

    # Push left command
    def push_left(self):
        # Variables
        moved, cur_score = False, 0
        shifted = []
        combs = []

        # Looping through and moving each number accordingly
        for r in range(4):
            spot, prev = 0, 0
            for c in range(4):
                current = self.grid[r, c]
                if current:  # if there's something there
                    if current == prev:  # if the number is the same as the previous
                        self.grid[r, spot - 1] += current  # combine into one
                        shifted.append((r, spot - 1))
                        combs.append((r, spot - 1))
                        cur_score += math.log2(current)  # update score
                        prev, moved = 0, True  # set variables
                    else:  # if the numbers are different
                        moved |= (spot != c)  # updates moved if unmoved
                        prev = self.grid[r, spot] = current  # update prev
                        if spot != c:
                            shifted.append((r, spot))
                        spot += 1  # increment i

            # Fill the remaining right part with 0
            while spot < 4:
                self.grid[r, spot] = 0
                spot += 1

        matches = self.check_neighbors(shifted)
        # print(cur_score, matches/MATCH_FIT_CONST)
        cur_score += matches

        # Return score or -1 if nothing moved
        return (cur_score if moved else -1), combs

    # Push right command
    def push_right(self):
        # Variables
        moved, cur_score = False, 0
        shifted = []
        combs = []

        # Looping through and moving each number accordingly
        for r in range(4):
            spot, prev = 3, 0
            for c in range(3, -1, -1):
                current = self.grid[r, c]
                if current:  # if there's something there
                    if current == prev:  # if the number is the same as the previous
                        self.grid[r, spot + 1] += current  # combine into one
                        shifted.append((r, spot + 1))
                        combs.append((r, spot + 1))
                        cur_score += math.log2(current)  # update score
                        prev, moved = 0, True  # set variables
                    else:  # if the numbers are different
                        moved |= (spot != c)  # updates moved if unmoved
                        prev = self.grid[r, spot] = current  # update prev
                        if spot != c:
                            shifted.append((r, spot))
                        spot -= 1  # decrement i

            # Fill the remaining top part with
            while 0 <= spot:
                self.grid[r, spot] = 0
                spot -= 1

        matches = self.check_neighbors(shifted)
        # print(cur_score, matches/MATCH_FIT_CONST)
        cur_score += matches

        # Return score or -1 if nothing moved
        return (cur_score if moved else -1), combs

    # Push up command
    def push_up(self):
        # Variables
        moved, cur_score = False, 0
        shifted = []
        combs = []

        # Looping through and moving each number accordingly
        for c in range(4):
            spot, prev = 0, 0
            for r in range(4):
                current = self.grid[r, c]
                if current:  # if there's something there
                    if current == prev:  # if the number is the same as the previous
                        self.grid[spot - 1, c] += current  # combine into one
                        shifted.append((spot - 1, c))
                        combs.append((spot - 1, c))
                        cur_score += math.log2(current)  # update score
                        prev, moved = 0, True  # set variables
                    else:  # if the numbers are different
                        moved |= (spot != r)  # updates moved if unmoved
                        prev = self.grid[spot, c] = current  # update prev
                        if spot != r:
                            shifted.append((spot, c))
                        spot += 1  # increment i

            # Fill the remaining bottom part with 0
            while spot < 4:
                self.grid[spot, c] = 0
                spot += 1

        matches = self.check_neighbors(shifted)
        # print(cur_score, matches/MATCH_FIT_CONST)
        cur_score += matches

        # Return score or -1 if nothing moved
        return (cur_score if moved else -1), combs

    # Push down command
    def push_down(self):
        # Variables
        moved, cur_score = False, 0
        shifted = []
        combs = []

        # Looping through and moving each number accordingly
        for c in range(4):
            spot, prev = 3, 0
            for r in range(3, -1, -1):
                current = self.grid[r, c]
                if current:  # if there's something there
                    if current == prev:  # if the number is the same as the previous
                        self.grid[spot + 1, c] += current  # combine into one
                        shifted.append((spot + 1, c))
                        combs.append((spot + 1, c))
                        cur_score += math.log2(current)  # update score
                        prev, moved = 0, True  # set variables
                    else:  # if the numbers are different
                        moved |= (spot != r)  # updates moved if unmoved
                        prev = self.grid[spot, c] = current  # update prev
                        if spot != r:
                            shifted.append((spot, c))
                        spot -= 1  # decrement i

            # Fill the remaining top part with
            while 0 <= spot:
                self.grid[spot, c] = 0
                spot -= 1

        matches = self.check_neighbors(shifted)
        # print(cur_score, matches/MATCH_FIT_CONST)
        cur_score += matches

        # Return score or -1 if nothing moved
        return (cur_score if moved else -1), combs

    def check_neighbors(self, shifted):
        temp = 0
        visited = []
        for r, c in shifted:
            val = self.grid[r, c]
            visited.append((r, c))
            if c > 0:
                if ((r, c - 1) not in visited) and (val == self.grid[r][c - 1]):
                    # print(f"({r}, {c}) and ({r}, {c - 1})")
                    temp += math.log2(val)
            if c < 3:
                if ((r, c + 1) not in visited) and (val == self.grid[r][c + 1]):
                    # print(f"({r}, {c}) and ({r}, {c + 1})")
                    temp += math.log2(val)
            if r > 0:
                if ((r - 1, c) not in visited) and (val == self.grid[r - 1][c]):
                    # print(f"({r}, {c}) and ({r - 1}, {c})")
                    temp += math.log2(val)
            if r < 3:
                if ((r + 1, c) not in visited) and (val == self.grid[r + 1][c]):
                    # print(f"({r}, {c}) and ({r + 1}, {c})")
                    temp += math.log2(val)
        return temp * MATCH_FIT_CONST

    # Putting new cells on the board
    def put_new_cell(self):
        i_s = []
        j_s = []

        # Find all empty slots
        for i in range(4):
            for j in range(4):
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


class OpGame:
    # Game constructor with default 4x4 board
    def __init__(self, start_two=True):
        self.grid = np.zeros(shape=(4, 4), dtype='uint16')
        self.score = 0
        self.end = False
        # Initialize board with 2 random numbers
        if start_two:
            self.put_new_cell()
            self.put_new_cell()

    def reset_game(self, start_two=True):
        self.grid = np.zeros(shape=(4, 4), dtype='uint16')
        self.score = 0
        self.end = False
        # Initialize board with 2 random numbers
        if start_two:
            self.put_new_cell()
            self.put_new_cell()

    # Flattens board to input into nn
    def flatten(self):
        flat = []
        for i in range(4):
            for j in range(4):
                curr = self.grid[i][j]
                if curr != 0:
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
        for i in range(4):
            for j in range(4):
                cur = self.grid[i, j]
                if cur != 0:
                    m.append((cur, i, j))
        m.sort(reverse=True)
        return m

    def maximum(self):
        m = 0
        for i in range(4):
            for j in range(4):
                cur = self.grid[i, j]
                m = max(m, cur)
        return m

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

    def next_right(self, r, c):
        while c < 3:
            rn = self.grid[r][c + 1]
            if rn == 0:
                c += 1
            else:
                return rn
        return 0

    def next_down(self, r, c):
        while r < 3:
            dn = self.grid[r + 1][c]
            if dn == 0:
                r += 1
            else:
                return dn
        return 0

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
        s = "\n"
        wall = "+------" * 4 + "+"
        s += wall + "\n"
        for i in range(4):
            insides = "|".join("{:^6}".format(self.grid[i, j]) for j in range(4))
            s += f"|{insides}|\n{wall}\n"
        return s

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

    def check_neighbors(self, shifted):
        temp = 0
        visited = []
        for r, c in shifted:
            val = self.grid[r, c]
            visited.append((r, c))
            if c > 0:
                if ((r, c - 1) not in visited) and (val == self.grid[r][c - 1]):
                    # print(f"({r}, {c}) and ({r}, {c - 1})")
                    temp += math.log2(val)
            if c < 3:
                if ((r, c + 1) not in visited) and (val == self.grid[r][c + 1]):
                    # print(f"({r}, {c}) and ({r}, {c + 1})")
                    temp += math.log2(val)
            if r > 0:
                if ((r - 1, c) not in visited) and (val == self.grid[r - 1][c]):
                    # print(f"({r}, {c}) and ({r - 1}, {c})")
                    temp += math.log2(val)
            if r < 3:
                if ((r + 1, c) not in visited) and (val == self.grid[r + 1][c]):
                    # print(f"({r}, {c}) and ({r + 1}, {c})")
                    temp += math.log2(val)
        return temp * MATCH_FIT_CONST

    # Putting new cells on the board
    def put_new_cell(self):
        i_s = []
        j_s = []

        # Find all empty slots
        for i in range(4):
            for j in range(4):
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
    g = ImprovedGame()
    score = random_play(g, True)
    print("Score: ", score)
