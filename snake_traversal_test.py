import math
import numpy as np
import random


def ordered():
    m = []
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            cur = grid[i, j]
            if cur != 0:
                m.append((cur, i, j))
    m.sort(reverse=True)
    return m


def corner_traverse(corner):
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

    vals = ordered()

    switch = False
    cont = True
    counter = 0
    for j in range(v_stuff[0][0], v_stuff[0][1], v_stuff[0][2]):
        for i in range(h_stuff[switch][0], h_stuff[switch][1], h_stuff[switch][2]):
            cur = grid[j][i]
            if cur == vals[counter][0]:
                if cur != 0:
                    nums1 += math.log2(cur)
                    counter += 1
                else:
                    cont = False
                    break
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
            cur = grid[j][i]
            if cur == vals[counter][0]:
                if cur != 0:
                    nums2 += math.log2(cur)
                    counter += 1
                else:
                    cont = False
                    break
            else:
                cont = False
                break
        if cont:
            switch = not switch
        else:
            break
    print(nums1, nums2)
    return max(nums1, nums2)


grid = np.zeros(shape=(4, 4), dtype='uint16')
vals = [2048, 1024, 512, 512, 1024, 4, 0, 256, 4, 2, 0, 0, 0, 0, 0, 0]
k = 0
for i in range(len(grid)):
    for j in range(len(grid[0])):
        # grid[i][j] = 2 ** random.randint(0, 13)
        grid[i][j] = vals[k]
        k += 1
print(grid)

num = corner_traverse(0)
print(num)
num = corner_traverse(1)
print(num)
num = corner_traverse(2)
print(num)
num = corner_traverse(3)
print(num)
