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


def grid_smoothness():
    smoothness = 0
    empty_sqs = 0
    matches = 0
    for i in range(4):
        for j in range(4):
            cur = grid[i][j]
            print(f"\n{cur}", end=": ")
            if cur:
                rn = next_right(i, j)
                print(rn, end=", ")
                if rn:
                    diff = math.log2(cur) - math.log2(rn)
                    if diff:
                        smoothness -= abs(diff)
                        print(f"\n\t{cur}, {rn}: {diff}", end="\t")
                    else:
                        matches += math.log2(cur)

                dn = next_down(i, j)
                print(dn)
                if dn:
                    diff = math.log2(cur) - math.log2(dn)
                    if diff:
                        smoothness -= abs(diff)
                        print(f"\n\t{cur}, {dn}: {diff}", end="\t")
                    else:
                        matches += math.log2(cur)
            else:
                print()
                empty_sqs += 1
    return smoothness, empty_sqs, matches


def next_right(r, c):
    while c < 3:
        rn = grid[r][c + 1]
        if rn == 0:
            c += 1
        else:
            return rn
    return 0


def next_down(r, c):
    while r < 3:
        dn = grid[r + 1][c]
        if dn == 0:
            r += 1
        else:
            return dn
    return 0


grid = np.zeros(shape=(4, 4), dtype='uint16')
vals = [2048, 64, 4, 512, 1024, 4, 0, 0, 4, 2, 0, 512, 0, 0, 2, 0]
k = 0
for i in range(len(grid)):
    for j in range(len(grid[0])):
        # grid[i][j] = 2 ** random.randint(0, 13)
        grid[i][j] = vals[k]
        k += 1
print(grid)

print(grid_smoothness())
