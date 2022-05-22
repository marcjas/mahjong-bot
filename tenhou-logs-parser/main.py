from copy import deepcopy
import re
from xmlrpc.client import Boolean
from TenhouDecoder import Game
import os
import sys 


class Dataset:
    def __init__(self, data):
        self._data = data
        self._players = data["players"]
        self._rounds = data["rounds"]

    def chi_model_data(self):
        pass

    def pon_model_data(self):
        states = []
        actions = []

        for index, round in enumerate(self._rounds):
            round_state = RoundState(round, player_scores)

            for event in round["events"]:
                round_state.update_state(event)

                if event["type"] == "Discard":
                    

            player_scores = self.update_player_scores(round, player_scores, round["deltas"])

        self.save_state_actions(states, actions)


        pass

    def kan_model_data(self):
        pass

    def riichi_model_data(self):
        states = []
        actions = []

        player_scores = [250, 250, 250, 250]

        for index, round in enumerate(self._rounds):
            round_state = RoundState(round, player_scores)

            for event in round["events"]:
                round_state.update_state(event)

                if event["type"] == "Draw":
                    player_idx = event["player"]

                    if round_state._player_states[player_idx].can_riichi():
                        test = round["round"]
                        turn = round_state._player_states[player_idx]._turn + 1

                        did_riichi = False
                        if player_idx in round_state._riichis_dict:
                            if round_state._riichis_dict[player_idx] == turn:
                                did_riichi = True
                        
                        states.append(round_state._player_states[player_idx])
                        actions.append(did_riichi)

                        print(f"{test} | player={player_idx}, turn={turn}, did_riichi={did_riichi}")

            player_scores = self.update_player_scores(round, player_scores, round["deltas"])

        self.save_state_actions(states, actions)


    def discard_model_data(self):
        states = []
        actions = []

        player_scores = [250, 250, 250, 250]

        for index, round in enumerate(self._rounds):
            round_state = RoundState(round, player_scores)

            for event in round["events"]:
                if event["type"] == "Discard":
                    player_idx = event["player"]
                    states.append(round_state._player_states[player_idx].to_model_input())
                    actions.append(event["tile"])

                round_state.update_state(event)


            player_scores = self.update_player_scores(round, player_scores, round["deltas"])

        self.save_state_actions(states, actions)


    def update_player_scores(self, round, old, deltas):
        new_scores = [a + b for a, b in zip(old, deltas)]

        if len(round["reaches"]) == 0:
            return new_scores

        for player_idx in round["reaches"]:
            new_scores[player_idx] -= 10

        return new_scores

    def save_state_actions(self, states, actions):
        pass


class RoundState:
    def __init__(self, round, player_scores):
        self._round = round 
        self._dealer = self._round["dealer"]
        self._player_scores = player_scores
        self._discarded_tiles = [[], [], [], []]
        self._open_tiles = [[], [], [], []]
        self._dora_indicators = []
        self._others_riichi_status = [0, 0, 0, 0]
        self._riichis = self._round["reaches"]
        self._riichi_turns = self._round["reach_turns"]
        self._riichis_dict = { self._riichis[i]: self._riichi_turns[i] for i in range(len(self._riichis)) }

        self._p0_state = PlayerObservableState(0, self._round, self._player_scores, self._discarded_tiles, self._open_tiles, self._dora_indicators, self.get_player_wind(0, self._dealer), self._others_riichi_status)
        self._p1_state = PlayerObservableState(1, self._round, self._player_scores, self._discarded_tiles, self._open_tiles, self._dora_indicators, self.get_player_wind(1, self._dealer), self._others_riichi_status)
        self._p2_state = PlayerObservableState(2, self._round, self._player_scores, self._discarded_tiles, self._open_tiles, self._dora_indicators, self.get_player_wind(2, self._dealer), self._others_riichi_status)
        self._p3_state = PlayerObservableState(3, self._round, self._player_scores, self._discarded_tiles, self._open_tiles, self._dora_indicators, self.get_player_wind(3, self._dealer), self._others_riichi_status)

        self._player_states = [self._p0_state, self._p1_state, self._p2_state, self._p3_state]

    def get_player_wind(self, player_idx, dealer_idx):
        winds = ["east", "south", "west", "north"]
        return winds[(player_idx - dealer_idx) % 4]

    def update_state(self, event):
        match event["type"]:
            case "Dora":
                dora_tile = event["tile"]
                valid_tile = bool(re.search("[a-z]", str(dora_tile)))
                if not valid_tile:
                    dora_tile = decode_tile(dora_tile)

                self._dora_indicators.append(dora_tile[0:2])

            case "Discard":
                player_idx = event["player"]
                self._player_states[player_idx].discard(event["tile"])

                player_turn = self._player_states[player_idx]._turn

                if player_idx in self._riichis_dict and player_turn == self._riichis_dict[player_idx]:
                    self._others_riichi_status[player_idx] = 1

            case "Draw":
                player_idx = event["player"]
                self._player_states[player_idx].draw(event["tile"])

            case "Call":
                player_idx = event["player"]
                self._player_states[player_idx].call(event["meld"])

        self.update_players_aka_dora_count_in_hand()

    def update_players_aka_dora_count_in_hand(self):
        self._p0_state.update_aka_dora_count_in_hand()
        self._p1_state.update_aka_dora_count_in_hand()
        self._p2_state.update_aka_dora_count_in_hand()
        self._p3_state.update_aka_dora_count_in_hand()
    
tiles = [
    "1m", "2m", "3m", "4m", 
    "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", 
    "5p", "6p", "7p", "8p", "9p", 
    "1s", "2s", "3s", "4s", 
    "5s", "6s", "7s", "8s", "9s", 
    "ew", "sw", "ww", "nw",       
    "wd", "gd", "rd"              
]

m_tiles = tiles[0:9]
p_tiles = tiles[9:18]
s_tiles = tiles[18:27]

class PlayerObservableState:
    def __init__(self, player_idx, round, player_scores, discarded_tiles, open_tiles, dora_indicators, wind, riichi_status):
        self._player_idx = player_idx 
        self._discarded_tiles = discarded_tiles
        self._open_tiles = open_tiles
        self._round_info = round["round"] # Tuple of string (name), int (honba count), int (leftover riichi sticks)
        self._turn = 0
        self._riichi_status = riichi_status
        self._is_open_hand = False

        """
        Features:
            - self._private_tiles               34 x 4 x 1
            - self._private_discarded tiles     34 x 30
            - self._others_discarded_tiles      34 x 30 x 3
            - self._private_open_tiles          --
            - self._dora_indicators             34 x 5
            - self._round_name                  one hot encoding 16length (4e, 4s, ...)
            - self._player_scores               scale from 0-1 where 0=0 and 1=100.000  [0.25, 0.25, 0.25, 0.25]
        """
        self._private_tiles = round["hands"][self._player_idx]
        self._private_discarded_tiles = self._discarded_tiles[self._player_idx]
        self._others_discarded_tiles = self._discarded_tiles[:self._player_idx] + self._discarded_tiles[self._player_idx + 1:]
        self._private_open_tiles = self._open_tiles[self._player_idx] 
        self._others_open_tiles = self._open_tiles[:self._player_idx] + self._open_tiles[self._player_idx + 1:]
        self._dora_indicators = dora_indicators
        self._round_name = self._round_info[0]
        self._player_scores = player_scores
        self._self_wind = wind
        self._aka_doras_in_hand = 0
        self._others_riichi_status = self._riichi_status[:self._player_idx] + self._riichi_status[self._player_idx + 1:]

        self.update_aka_dora_count_in_hand()

    def update_aka_dora_count_in_hand(self):
        aka_doras = ["5m0", "5s0", "5p0"]

        aka_dora_count_private = sum([1 if (i in aka_doras) else 0 for i in self._private_tiles])
        aka_dora_count_open = sum([1 if (i in aka_doras) else 0 for i in self._private_open_tiles])

        self._aka_doras_in_hand =  aka_dora_count_private + aka_dora_count_open

    def get_score(self):
        return self._player_scores[self._player_idx]

    def discard(self, tile):
        self._private_tiles.remove(tile)
        self._discarded_tiles[self._player_idx].append(tile)

        self._turn += 1

    def draw(self, tile):
        self._private_tiles.append(tile)

    def call(self, meld):
        tiles = meld["tiles"]

        if meld["type"] == "kan":
            self._is_open_hand = True

        for tile in tiles:
            if tile in self._private_tiles:
                self._private_tiles.remove(tile)

        self._open_tiles[self._player_idx].append(tiles)

    def can_riichi(self) -> Boolean:
        if self._player_scores[self._player_idx] < 10:
            return False

        if self._riichi_status[self._player_idx] == 1:
            return False

        if self._is_open_hand == True:
            return False

        # easier to work with
        transformed_hand = self.transform_hand(deepcopy(self._private_tiles))
        # print(transformed_hand)

        m_tiles = transformed_hand[0:9]
        p_tiles = transformed_hand[9:18]
        s_tiles = transformed_hand[18:27]
        w_tiles = transformed_hand[27:31]
        d_tiles = transformed_hand[31:34]

        m_combinations = self.find_suit_combinations(m_tiles)
        p_combinations = self.find_suit_combinations(p_tiles)
        s_combinations = self.find_suit_combinations(s_tiles)
        w_combinations = self.find_honor_combinations(w_tiles)
        d_combinations = self.find_honor_combinations(d_tiles)

        combinations = [m_combinations, p_combinations, s_combinations, w_combinations, d_combinations]
        counts = [0, 0, 0, 0]

        for combination in combinations:
            for idx, count in enumerate(combination):
                counts[idx] += count 

        # check for chiitoi
        pair_count = 0
        for tile in transformed_hand:
            if tile >= 2:
                pair_count += 1
        if pair_count == 6:
            return True

        # check for kokushi musou
        # 13wait OR 11 + pair 
        # sum of specific indices == 13
        # TODO

        # (4, 0, 0, 2) eg. 123 123 444 555 2 ew
        # (4, 0, 1, 0) eg. 123 123 444 555 23
        # (3, 2, 0, 1) eg. 123 123 123 44 55 ew 
        # (3, 1, 1, 1) eg. 123 123 444 55 23 ew
        if counts == [4, 0, 0, 2] \
            or counts == [4, 0, 1, 0] \
            or counts == [3, 2, 0, 1] \
            or counts == [3, 1, 1, 1]:
                return True

        return False

    def find_suit_combinations(self, tiles):
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
            tiles[idx] -= count 
            tiles[idx + 1] -= count
            tiles[idx + 2] -= count
            
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

    def find_honor_combinations(self, tiles):
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
        
    def transform_hand(self, hand):
        new_hand = [0] * 34

        mapped_hand = list(map(lambda x: x[0:2], hand))

        for tile in mapped_hand:
            idx = tiles.index(tile)
            new_hand[idx] += 1


        return new_hand


    def __str__(self):
        return f"""
_private_tiles:
    {self._private_tiles}

_private_open_tiles: 
    {self._private_open_tiles}    

_private_discarded_tiles:
    {self._private_discarded_tiles}

_others_discarded_tiles:
    {self._others_discarded_tiles}

_others_open_tiles: 
    {self._others_open_tiles}

_dora_indicators:
    {self._dora_indicators}

_round_name:
    {self._round_name}

_player_scores:
    {self._player_scores}

_self_wind: 
    {self._self_wind}

_aka_doras_in_hand:
    {self._aka_doras_in_hand}

_others_riichi_status:
    {self._others_riichi_status}
        """

    def to_model_input(self):
        return self


def decode_tile(tile_code):
    UNICODE_TILES = """
        🀇 🀈 🀉 🀊 🀋 🀌 🀍 🀎 🀏
        🀙 🀚 🀛 🀜 🀝 🀞 🀟 🀠 🀡
        🀐 🀑 🀒 🀓 🀔 🀕 🀖 🀗 🀘
        🀀 🀁 🀂 🀃 
        🀆 🀅 🀄
    """.split()

    TILES = """
        1m 2m 3m 4m 5m 6m 7m 8m 9m
        1p 2p 3p 4p 5p 6p 7p 8p 9p
        1s 2s 3s 4s 5s 6s 7s 8s 9s
        ew sw ww nw
        wd gd rd
    """.split()

    return TILES[tile_code // 4] + str(tile_code % 4)

def main(logs_dir = "../tenhou-logs/xml-logs"):
    game = Game('DEFAULT')

    log_paths = [logs_dir + f for f in os.listdir(logs_dir)]

    for log_path in log_paths:
        print(log_path)

        game.decode(open(log_path))

        game_data = game.asdata()

        model_input = Dataset(game_data) 

        # model_input.discard_model_data()
        model_input.riichi_model_data()

        break


if __name__ == '__main__':
    args = sys.argv[1:]

    if len(args) == 0:
        main()
    elif len(args) == 1:
        main(args[0])
    else:
        print("invalid command")