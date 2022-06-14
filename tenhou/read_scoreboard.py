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

    scoreboard_reader = ScoreboardReader(scoreboard=scoreboard)

    turn_player = scoreboard_reader.get_turn_player()
    print("Turn player:", ["player", "right", "opposing", "left"][turn_player])

    dealer = scoreboard_reader.get_dealer()
    print("Dealer:", ["player", "right", "opposing", "left"][dealer])

    round_wind = scoreboard_reader.get_round_wind()
    round_number = scoreboard_reader.get_round_number()
    print("Round:", ["east", "south", "west", "north"][round_wind], round_number)

    player_winds = scoreboard_reader.get_player_winds()
    print("Player Winds:", [["east", "south", "west", "north"][w] for w in player_winds])

    player_scores = scoreboard_reader.get_player_scores()
    print("Scores:", player_scores)

class ScoreboardReader:
    def __init__(self, scoreboard=None):
        if scoreboard is not None:
            self.read(scoreboard)
        else:
            self.scoreboard = None

    def read(self, scoreboard):
        scoreboard = cv2.resize(scoreboard, (expected_width, expected_height))
        self.scoreboard = scoreboard

    def get_turn_player(self):
        points = [
            [round(self.scoreboard.shape[1] / 2), self.scoreboard.shape[0] - 1],
            [self.scoreboard.shape[1] - 1, round(self.scoreboard.shape[0] / 2)],
            [round(self.scoreboard.shape[1] / 2), 0],
            [0, round(self.scoreboard.shape[0] / 2)]
        ]

        for i in range(4):
            point = self.scoreboard[points[i][1], points[i][0]]
            if point[1] > point[0] + point[2]:
                return i

        return None

    def get_dealer(self):
        points = [
            [26, 102],
            [136, 104],
            [118, 12],
            [10, 10]
        ]

        for i in range(4):
            x = points[i][0]
            y = points[i][1]
            j = 40
            while j > 0:
                j -= 1
                x += [1, 0, -1, 0][i]
                y += [0, -1, 0, 1][i]
                point = self.scoreboard[y, x]
                if sum(point) > 120:
                    j = min([j, 6])
                if sum(point) > 360:
                    return i

        return None

    def get_round_wind(self):
        x = 40
        y = 52
        for i in range(70):
            point = self.scoreboard[y, x]
            if point[0] > int(point[1]) + int(point[2]):
                if i < 10:
                    return 2
                else:
                    return 0
            elif point[2] > int(point[0]) + int(point[1]):
                return 1
            x += 1

        return 0

    def get_round_number(self):
        x = 75
        y = 41
        consecutive = 0
        for i in range(20):
            point = self.scoreboard[y, x]
            if point[2] > int(point[0]) + int(point[1]):
                return 4
            elif consecutive > 6:
                return 1
            elif sum(point) > min_color:
                consecutive += 1
            elif consecutive > 0:
                if consecutive == 1:
                    return 3
                else:
                    return 2
            y += 1
        return 1

    def get_player_winds(self):
        player_wind = self.get_player_wind()
        winds = []
        for i in range(4):
            winds.append((player_wind + i) % 4)
        return winds

    def get_player_wind(self):
        dealer = self.get_dealer()
        player_wind = (-dealer) % 4
        return player_wind

    def get_player_scores(self):
        scores = [
            self.get_player_score(player=0),
            self.get_player_score(player=1),
            self.get_player_score(player=2),
            self.get_player_score(player=3)
        ]
        return scores

    def get_player_score(self, player=0):
        x = [102, 131, 44, 18][player]
        y = [98, 30, 18, 86][player]

        digits = []
        distance = 0
        drought = 0
        while len(digits) < 3 and distance < 100:
            if sum(self.scoreboard[y, x]) > min_color:
                points = self.get_connected_points(self.scoreboard, x, y)
                if player == 0:
                    x = min([p[0] for p in points])
                elif player == 1:
                    y = max([p[1] for p in points])
                elif player == 2:
                    x = max([p[0] for p in points])
                elif player == 3:
                    y = min([p[1] for p in points])
                if player % 2 == 1:
                    points = [(p[1], p[0]) for p in points]
                digits.append(points)
                drought = 0
            if player == 0:
                x -= 1
            elif player == 1:
                y += 1
            elif player == 2:
                x += 1
            elif player == 3:
                y -= 1
            distance += 1
            drought += 1
            if drought > 11 and len(digits) > 0:
                break

        if len(digits) == 0:
            return 0

        score = 0
        for i, digit in enumerate(digits):
            score += self.read_digit(digit, player=player, pos=i) * 10**i
        return score

    def read_digit(self, points, player=0, pos=0):
        min_x = min([p[0] for p in points])
        min_y = min([p[1] for p in points])
        points = [(p[0] - min_x, p[1] - min_y) for p in points]
        min_x = 0
        min_y = 0
        max_x = max([p[0] for p in points])
        max_y = max([p[1] for p in points])
        if player == 1:
            points = [(max_x - p[0], p[1]) for p in points]
        elif player == 2:
            points = [(max_x - p[0], max_y - p[1]) for p in points]
        elif player == 3:
            points = [(p[0], max_y - p[1]) for p in points]

        image = np.zeros((20, 20), np.uint8)
        for p in points:
            image[p[1], p[0]] = 255

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
        subpoints2 = self.get_connected_points(image, 4, 6)
        if len(subpoints) == len(subpoints2):
            return 8

        # Check for 6
        subpoints = filter_points(points, lambda p: p[1] < max_y - 3 or p[1] > max_y - 3 or p[0] < 4)
        image = np.zeros((17, 12, 3), np.uint8)
        for p in subpoints:
            image[p[1], p[0], 0] = 255
            image[p[1], p[0], 1] = 255
            image[p[1], p[0], 2] = 255
        subpoints2 = self.get_connected_points(image, 1, 6)
        if len(subpoints) == len(subpoints2):
            return 6

        # Check for 9
        subpoints = filter_points(points, lambda p: p[1] < 3 or p[1] > 3 or p[0] < 4)
        image = np.zeros((17, 12, 3), np.uint8)
        for p in subpoints:
            image[p[1], p[0], 0] = 255
            image[p[1], p[0], 1] = 255
            image[p[1], p[0], 2] = 255
        subpoints2 = self.get_connected_points(image, 1, 4)
        if len(subpoints) == len(subpoints2) and len(points) >= 60:
            return 9

        subpoints2 = self.get_connected_points(image, 4, 1)
        if len(subpoints) == len(subpoints2):
            return 5
        
        return 3


    def get_connected_points(self, image, x, y):
        points = []
        def DFS(points, x, y):
            points.append((x, y))
            neighbors = [
                (1, 0), (0, 1), (-1, 0), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ]
            for n in neighbors:
                point = (x - n[0], y - n[1])
                if point[0] < image.shape[1] and point[1] < image.shape[0] and \
                point[0] >= 0 and point[1] >= 0:
                    if sum(image[point[1], point[0]]) > min_color and point not in points:
                        DFS(points, point[0], point[1])
        DFS(points, x, y)
        return points

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Needs to receive the name of a file from the scoreboards directory.")
        print("The filename should be without the directory and extension.")
        print("Usage: python tenhou/read_scoreboard.py <filename>")
        sys.exit(-1)
    main(sys.argv[1])