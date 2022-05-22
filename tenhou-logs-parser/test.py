"""
if counts == [4, 0, 0, 2] \
    or counts == [4, 0, 1, 0] \
    or counts == [3, 2, 0, 1] \
    or counts == [3, 0, 1, 1]:
        return True
"""

def find_suit_combinations(tiles):
    mentsu_count = 0
    pair_count = 0
    taatsu_count = 0
    isolated_count = 0

    # check koutsu (triplet)
    for idx in range(len(tiles)):
        if tiles[idx] >= 3:
            mentsu_count += 1
            tiles[idx] -= 3

    # check shuntsu (sequence)
    for idx in range(len(tiles) - 2):
        if tiles[idx] == 0 or tiles[idx + 1] == 0 or tiles[idx + 2] == 0:
            continue

        sum = tiles[idx] + tiles[idx + 1] + tiles[idx + 2]
        count = sum // 3

        mentsu_count += count
        # tiles = [cnt - 1 for cnt in tiles[idx:idx+3]]
        tiles[idx] -= count 
        tiles[idx + 1] -= count
        tiles[idx + 2] -= count
        
    # # check taatsu (unfinished sequence eg. 1-2, 1-3, 2-3)
    # for idx in range(len(tiles) - 2):
    #     sum = tiles[idx] + tiles[idx + 1] + tiles[idx + 2]

    #     if sum == 2:
    #         taatsu_count += 1
    #         if tiles[idx] == 1:
    #             tiles[idx] -= 1

    #         if tiles[idx + 1] == 1:
    #             tiles[idx + 1] -= 1

    #         if tiles[idx + 2] == 1:
    #             tiles[idx + 2] -= 1

    # # check pair 
    # for idx in range(len(tiles)):
    #     if tiles[idx] == 2:
    #         pair_count += 1
    #         tiles[idx] -= 2

    # check pair 
    for idx in range(len(tiles)):
        if tiles[idx] == 2:
            pair_count += 1
            tiles[idx] -= 2

    # check taatsu (unfinished sequence eg. 1-2, 1-3, 2-3)
    for idx in range(len(tiles) - 2):
        sum = tiles[idx] + tiles[idx + 1] + tiles[idx + 2]

        if sum == 2:
            taatsu_count += 1
            if tiles[idx] == 1:
                tiles[idx] -= 1

            if tiles[idx + 1] == 1:
                tiles[idx + 1] -= 1

            if tiles[idx + 2] == 1:
                tiles[idx + 2] -= 1

    # check isolated 
    sum = 0
    for cnt in tiles:
        sum += cnt

    isolated_count += sum

    return (mentsu_count, pair_count, taatsu_count, isolated_count)

def find_honor_combinations(tiles):
    mentsu_count = 0
    pair_count = 0
    isolated_count = 0

    # check koutsu (triplet)
    for idx in range(len(tiles)):
        if tiles[idx] >= 3:
            mentsu_count += 1
            tiles[idx] -= 3

    # check pair 
    for idx in range(len(tiles)):
        if tiles[idx] == 2:
            pair_count += 1
            tiles[idx] -= 2

    # check isolated 
    sum = 0
    for cnt in tiles:
        sum += cnt

    isolated_count += sum

    return (mentsu_count, pair_count, 0, isolated_count)

if __name__ == "__main__":
    # m_tiles = [0, 0, 0, 3, 0, 0, 1, 1, 1]   # (2, 0, 0, 0)
    # p_tiles = [0, 0, 0, 0, 0, 1, 1, 0, 2]   # (0, 1, 1, 0)
    # s_tiles = [0, 1, 2, 1, 0, 0, 0, 0, 0]   # (1, 0, 0, 1)
    # w_tiles = [0, 0, 0, 0]
    # d_tiles = [0, 0, 0]
    # # (3, 1, 1, 1)
    # # 123 123 123 56 88 0
    # # kan riichi

    # m_tiles = [0, 0, 0, 0, 0, 0, 0, 0, 0]   # (0, 0, 0, 0)
    # p_tiles = [0, 0, 0, 1, 1, 1, 1, 1, 1]   # (2, 0, 0, 0)
    # s_tiles = [0, 1, 0, 1, 0, 1, 1, 1, 0]   # (1, 0, 1, 0)
    # w_tiles = [0, 0, 2, 1]                  # (0, 1, 0, 1)
    # d_tiles = [0, 0, 0]
    # # (3, 1, 1, 1)
    # # 456 789 24 678 22 nw
    # # kan riichi

    m_tiles = [0, 0, 2, 1, 0, 2, 0, 0, 0]   # (0, 2, 0, 1)
    p_tiles = [0, 0, 2, 2, 3, 0, 0, 0, 0]   # (1, 2, 0, 0)
    s_tiles = [0, 0, 0, 0, 0, 2, 0, 0, 0]   # (0, 1, 0, 0)
    w_tiles = [0, 0, 0, 0]
    d_tiles = [0, 0, 0]
    # (1, 5, 0, 1)
    # 334 66 334455 66 5

    m_combinations = find_suit_combinations(m_tiles)
    p_combinations = find_suit_combinations(p_tiles)
    s_combinations = find_suit_combinations(s_tiles)
    w_combinations = find_honor_combinations(w_tiles)
    d_combinations = find_honor_combinations(d_tiles)

    combinations = [m_combinations, p_combinations, s_combinations, w_combinations, d_combinations]
    counts = [0, 0, 0, 0]

    for combination in combinations:
        for idx, count in enumerate(combination):
            counts[idx] += count 

    print(counts)