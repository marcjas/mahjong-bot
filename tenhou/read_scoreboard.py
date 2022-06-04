import cv2
import math
import read_tile
import sys
import numpy as np

expected_width  = 150
expected_height = 117
min_color = 240

def main(filename):
    scoreboard = cv2.imread(f"scoreboards/{filename}.png", cv2.IMREAD_COLOR)
    scoreboard = cv2.resize(scoreboard, (150, 117), interpolation=cv2.INTER_LINEAR)

    turn_player = get_turn_player(scoreboard)
    print("Turn player:", ["player", "right", "opposing", "left"][turn_player])

    scores = get_scores(scoreboard)
    print("Scores:", scores)

def get_scores(scoreboard):
    scores = [
        get_player_score(scoreboard, 102, 98, owner=0),
        get_player_score(scoreboard, 131, 30, owner=1),
        get_player_score(scoreboard, 44, 18, owner=2),
        get_player_score(scoreboard, 18, 86, owner=3)
    ]
    return scores

def get_player_score(scoreboard, x, y, owner=0):
    digits = []
    distance = 0
    drought = 0
    while len(digits) < 3 and distance < 100:
        if sum(scoreboard[y, x]) > min_color:
            points = get_connected_points(scoreboard, x, y)
            if owner == 0:
                x = min([p[0] for p in points])
            elif owner == 1:
                y = max([p[1] for p in points])
            elif owner == 2:
                x = max([p[0] for p in points])
            elif owner == 3:
                y = min([p[1] for p in points])
            if owner % 2 == 1:
                points = [(p[1], p[0]) for p in points]
            digits.append(points)
            drought = 0
        if owner == 0:
            x -= 1
        elif owner == 1:
            y += 1
        elif owner == 2:
            x += 1
        elif owner == 3:
            y -= 1
        distance += 1
        drought += 1
        if drought > 11 and len(digits) > 0:
            break

    if len(digits) == 0:
        return 0

    score = 0
    for i, digit in enumerate(digits):
        score += read_digit(digit, owner=owner, pos=i) * 10**i
    return score

def read_digit(points, owner=0, pos=0):
    min_x = min([p[0] for p in points])
    min_y = min([p[1] for p in points])
    points = [(p[0] - min_x, p[1] - min_y) for p in points]
    min_x = 0
    min_y = 0
    max_x = max([p[0] for p in points])
    max_y = max([p[1] for p in points])
    if owner == 1:
        points = [(max_x - p[0], p[1]) for p in points]
    elif owner == 2:
        points = [(max_x - p[0], max_y - p[1]) for p in points]
    elif owner == 3:
        points = [(p[0], max_y - p[1]) for p in points]

    image = np.zeros((20, 20), np.uint8)
    for p in points:
        image[p[1], p[0]] = 255
    #cv2.imwrite(f"digit_{owner}_{pos}.png", image)

    # Check for 1
    dims = get_point_list_dimensions(points)
    if dims[0] < 8:
        return 1

    # Check for 0
    subpoints = filter_points(points, lambda p: p[0] == max_x // 2 and p[1] > 2 and p[1] < max_y - 2)
    if len(subpoints) == 0:
        return 0

    # Check for 2
    subpoints = filter_points(points, lambda p: p[1] >= 14)
    dims = get_point_list_dimensions(subpoints)
    if dims[0] >= 7:
        subpoints = filter_points(points, lambda p: p[1] == 7 and p[0] < 3)
        if len(subpoints) == 0:
            return 2

    # Check for 7
    subpoints = filter_points(points, lambda p: p[1] < 1)
    dims = get_point_list_dimensions(subpoints)
    if dims[0] >= 8:
        return 7

    # Check for 4
    subpoints = filter_points(points, lambda p: p[0] == 6)
    min_y = min([p[1] for p in subpoints])
    max_y = max([p[1] for p in subpoints])
    solid = True
    for y in range(min_y, max_y+1):
        if (6, y) not in subpoints:
            solid = False
            break
    if solid:
        return 4

    # Check for 8
    min_y = min([p[1] for p in subpoints])
    max_y = max([p[1] for p in subpoints])
    subpoints = filter_points(points, lambda p: (p[1] > 2 and p[1] < max_y - 2) or p[0] > max_x // 2 or p[0] < max_x // 2)
    image = np.zeros((17, 12, 3), np.uint8)
    for p in subpoints:
        image[p[1], p[0], 0] = 255
        image[p[1], p[0], 1] = 255
        image[p[1], p[0], 2] = 255
    subpoints2 = get_connected_points(image, 4, 6)
    if len(subpoints) == len(subpoints2):
        return 8

    # Check for 6
    subpoints = filter_points(points, lambda p: p[1] < max_y - 3 or p[1] > max_y - 3 or p[0] < 4)
    image = np.zeros((17, 12, 3), np.uint8)
    for p in subpoints:
        image[p[1], p[0], 0] = 255
        image[p[1], p[0], 1] = 255
        image[p[1], p[0], 2] = 255
    subpoints2 = get_connected_points(image, 1, 6)
    if len(subpoints) == len(subpoints2):
        return 6

    # Check for 9
    subpoints = filter_points(points, lambda p: p[1] < 3 or p[1] > 3 or p[0] < 4)
    image = np.zeros((17, 12, 3), np.uint8)
    for p in subpoints:
        image[p[1], p[0], 0] = 255
        image[p[1], p[0], 1] = 255
        image[p[1], p[0], 2] = 255
        #cv2.imwrite(f"digit_{owner}_{pos}.png", image)
    subpoints2 = get_connected_points(image, 1, 4)
    if len(subpoints) == len(subpoints2) and len(points) >= 60:
        return 9

    subpoints2 = get_connected_points(image, 4, 1)
    if len(subpoints) == len(subpoints2):
        return 5
    
    return 3

def filter_points(points, func):
    return [p for p in points if func(p)]

def get_point_list_dimensions(points):
    if len(points) == 0:
        return (0, 0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return (width, height)

def get_connected_points(scoreboard, x, y):
    points = []
    def DFS(points, x, y):
        points.append((x, y))
        neighbors = [
            (1, 0), (0, 1), (-1, 0), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        for n in neighbors:
            point = (x - n[0], y - n[1])
            if point[0] < scoreboard.shape[1] and point[1] < scoreboard.shape[0] and \
               point[0] >= 0 and point[1] >= 0:
                if sum(scoreboard[point[1], point[0]]) > min_color and point not in points:
                    DFS(points, point[0], point[1])
    DFS(points, x, y)
    return points

def get_turn_player(scoreboard):
    points = [
        [round(scoreboard.shape[1] / 2), scoreboard.shape[0] - 1],
        [scoreboard.shape[1] - 1, round(scoreboard.shape[0] / 2)],
        [round(scoreboard.shape[1] / 2), 0],
        [0, round(scoreboard.shape[0] / 2)]
    ]

    for i in range(4):
        point = scoreboard[points[i][1], points[i][0]]
        if point[1] > point[0] + point[2]:
            return i

    print("Found no turn player.")
    sys.exit(-1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Needs to receive the name of a file from the scoreboards directory.")
        print("The filename should be without the directory and extension.")
        print("Usage: python tenhou/read_scoreboard.py <filename>")
        sys.exit(-1)
    main(sys.argv[1])