from game import AdversarialGame
from sys import maxsize as MAX_INT


# Find all possible moves for player
def get_max_moves(grid):
    moves = []
    if check_left(grid):
        moves.append(0)
    if check_up(grid):
        moves.append(1)
    if check_right(grid):
        moves.append(2)
    if check_down(grid):
        moves.append(3)
    return moves


# Check if left is possible move
def check_left(grid):
    for r in range(4):
        k = -1
        for c in range(3, -1, -1):
            if grid.grid[r][c] > 0:
                k = c
                break
        if k > -1:
            for c in range(k, 0, -1):
                if grid.grid[r][c - 1] == 0 or grid.grid[r][c] == grid.grid[r][c - 1]:
                    return True
    return False


# Check if up is possible move
def check_up(grid):
    for c in range(4):
        k = -1
        for r in range(3, -1, -1):
            if grid.grid[r][c] > 0:
                k = r
                break
        if k > -1:
            for r in range(k, 0, -1):
                if grid.grid[r - 1][c] == 0 or grid.grid[r][c] == grid.grid[r - 1][c]:
                    return True
    return False


# Check if right is possible move
def check_right(grid):
    for r in range(4):
        k = -1
        for c in range(4):
            if grid.grid[r][c] > 0:
                k = c
                break
        if k > -1:
            for c in range(k, 3):
                if grid.grid[r][c + 1] == 0 or grid.grid[r][c] == grid.grid[r][c + 1]:
                    return True
    return False


# Check if down is possible move
def check_down(grid):
    for c in range(4):
        k = -1
        for r in range(4):
            if grid.grid[r][c] > 0:
                k = r
                break
        if k > -1:
            for r in range(k, 3):
                if grid.grid[r + 1][c] == 0 or grid.grid[r][c] == grid.grid[r + 1][c]:
                    return True
    return False


# Find all placements for board
def get_min_moves(grid):
    moves = []
    empty = grid.get_empty()
    for r, c in empty:
        moves.append((r, c, 2))
        moves.append((r, c, 4))
    return moves


# Check if game is in terminal state
def is_terminal(grid, player):
    if player:
        if check_left(grid):
            return False
        if check_up(grid):
            return False
        if check_right(grid):
            return False
        if check_down(grid):
            return False
        return True
    else:
        empty = grid.get_empty()
        if empty:
            return False
        else:
            return True


# Check which move was performed
def get_move_to(grid, child):
    if check_left(grid):
        g = AdversarialGame(start_two=False, matrix=grid.get_grid())
        g.push_left()
        if g == child:
            return 0
    if check_up(grid):
        g = AdversarialGame(start_two=False, matrix=grid.get_grid())
        g.push_up()
        if g == child:
            return 1
    if check_right(grid):
        g = AdversarialGame(start_two=False, matrix=grid.get_grid())
        g.push_right()
        if g == child:
            return 2
    return 3


# Maximize of minimax algorithm
def maximize(grid, alpha, beta, depth):
    # Keep track of best options
    best_grid = None
    best_fitness = -1

    # Exit if it is over
    if depth == 0 or is_terminal(grid, True):
        return None, grid.fitness()

    # Check over every child
    for child in get_max_moves(grid):
        # Duplicate game and move
        g = AdversarialGame(start_two=False, matrix=grid.get_grid())
        g.game_move(child)

        # Check all minimize options
        _, fitness = minimize(g, alpha, beta, depth - 1)
        # Update if new maximum
        if fitness > best_fitness:
            best_grid, best_fitness = g, fitness

        # Alpha-beta pruning to speed up process
        if best_fitness >= beta:
            break
        if best_fitness > alpha:
            alpha = best_fitness

    # Returns best options
    return best_grid, best_fitness


# Minimize of minimax algorithm
def minimize(grid, alpha, beta, depth):
    # Keep track of best options
    worst_grid = None
    worst_fitness = MAX_INT

    # Exit if it is over
    if depth == 0 or is_terminal(grid, False):
        return None, grid.fitness()

    # Check over every child
    for child in get_min_moves(grid):
        # Duplicate game and place
        g = AdversarialGame(start_two=False, matrix=grid.get_grid())
        g.place_tile(child[0], child[1], child[2])

        # Check all maximize options
        _, fitness = maximize(g, alpha, beta, depth - 1)
        # Update if new minimum
        if fitness < worst_fitness:
            worst_grid, worst_fitness = g, fitness

        # Alpha-beta pruning to speed up process
        if worst_fitness <= alpha:
            break
        if worst_fitness < beta:
            beta = worst_fitness

    # Returns worst options
    return worst_grid, worst_fitness


def get_best_move(grid, depth=5):
    child, _ = maximize(AdversarialGame(start_two=False, matrix=grid.get_grid()), -1, MAX_INT, depth)
    return get_move_to(grid, child)
