import argparse
from copy import deepcopy
import re
from typing import Dict
from xmlrpc.client import Boolean
from TenhouDecoder import Game
import os
import sys
import logging
import csv
import time


LOGGING_LEVEL = logging.INFO

GET_CHII_MODEL_INPUT        = True 
GET_PON_MODEL_INPUT         = True 
GET_KAN_MODEL_INPUT         = True 
GET_RIICHI_MODEL_INPUT      = True 
GET_DISCARD_MODEL_INPUT     = True

class GameLogParser:
    states = [[], [], [], [], []]
    actions = [[], [], [], [], []]

    def __init__(self, data: Dict):
        self._data = data
        self._players = data["players"]
        self._rounds = data["rounds"]

    def get_nn_input(self, export_to_csv=True):
        player_scores = [250, 250, 250, 250]

        for round in self._rounds:
            round_state = RoundState(round, player_scores)
            for event_idx, event in enumerate(round["events"]):
                if GET_DISCARD_MODEL_INPUT:
                    self.get_discard_model_data(round_state, round, event)

                round_state.update_state(event)

                if GET_CHII_MODEL_INPUT:
                    self.get_chii_model_data(round_state, round, event, event_idx)

                if GET_PON_MODEL_INPUT:
                    self.get_pon_model_data(round_state, round, event, event_idx)

                if GET_KAN_MODEL_INPUT:
                    self.get_kan_model_data(round_state, round, event, event_idx)

                if GET_RIICHI_MODEL_INPUT:
                    self.get_riichi_model_data(round_state, round, event)

            player_scores = self.update_player_scores(round, player_scores, round["deltas"])

            logging.debug(f'{round["round"]} | scores={player_scores}')
        
        if export_to_csv:
            self.export_to_csv()

    def get_chii_model_data(self, round_state, round, event, event_idx):
        if event["type"] == "Discard":
            discard_player_idx = event["player"]
            player_idx = ((discard_player_idx + 1) + 4) % 4
            
            if round_state._others_riichi_status[player_idx] == 1:
                return 

            if round_state._player_states[player_idx].can_chii(event["tile"]):
                did_chii = False
                next_event_idx = event_idx + 1

                try:
                    next_event = round["events"][next_event_idx]
                except:
                    return 

                if next_event["type"] == "Call" and next_event["meld"]["type"] == "chi" and next_event["player"] == player_idx:
                    did_chii = True

                self.states[0].append(round_state._player_states[player_idx])
                self.actions[0].append(did_chii)

                logging.debug(f'{round["round"]} | player={player_idx}, turn={round_state._player_states[player_idx]._turn}, did_chii={did_chii}')

    def get_pon_model_data(self, round_state, round, event, event_idx):
        if event["type"] == "Discard":
            discard_player_idx = event["player"]
            for player_idx in [x for x in range(4) if x != discard_player_idx]:
                if round_state._others_riichi_status[player_idx] == 1:
                    continue
                
                if round_state._player_states[player_idx].can_pon(event["tile"]):
                    did_pon = False
                    next_event_idx = event_idx + 1

                    try:
                        next_event = round["events"][next_event_idx]
                    except:
                        break 

                    if next_event["type"] == "Call" and next_event["meld"]["type"] == "pon" and next_event["player"] == player_idx:
                        did_pon = True

                    self.states[1].append(round_state._player_states[player_idx])
                    self.actions[1].append(did_pon)

                    logging.debug(f'{round["round"]} | player={player_idx}, turn={round_state._player_states[player_idx]._turn}, did_pon={did_pon}')

    def get_kan_model_data(self, round_state, round, event, event_idx):
        if event["type"] == "Draw":
            player_idx = event["player"]
            if round_state._player_states[player_idx].can_chakan():
                did_chakan = False
                next_event_idx = event_idx + 1

                try:
                    next_event = round["events"][next_event_idx]
                except:
                    return 

                if next_event["type"] == "Call" and next_event["meld"]["type"] == "chakan" and next_event["player"] == player_idx:
                    did_chakan = True

                self.states[2].append(round_state._player_states[player_idx])
                self.actions[2].append(did_chakan)

                logging.debug(f'{round["round"]} | player={player_idx}, turn={round_state._player_states[player_idx]._turn}, did_chakan={did_chakan}')

        if event["type"] == "Discard":
            discard_player_idx = event["player"]
            for player_idx in [x for x in range(4) if x != discard_player_idx]:
                if round_state._others_riichi_status[player_idx] == 1:
                    continue

                if round_state._player_states[player_idx].can_kan(event["tile"]):
                    did_kan = False
                    next_event_idx = event_idx + 1

                    try:
                        next_event = round["events"][next_event_idx]
                    except:
                        return 

                    if next_event["type"] == "Call" and next_event["meld"]["type"] == "kan" and next_event["player"] == player_idx:
                        did_kan = True

                    self.states[2].append(round_state._player_states[player_idx])
                    self.actions[2].append(did_kan)

                    logging.debug(f'{round["round"]} | player={player_idx}, turn={round_state._player_states[player_idx]._turn}, did_kan={did_kan}')

    def get_riichi_model_data(self, round_state, round, event):
        if event["type"] == "Draw":
            player_idx = event["player"]

            if round_state._player_states[player_idx].can_riichi():
                turn = round_state._player_states[player_idx]._turn + 1

                did_riichi = False
                if player_idx in round_state._riichis_dict:
                    if round_state._riichis_dict[player_idx] == turn:
                        did_riichi = True
                
                self.states[3].append(round_state._player_states[player_idx])
                self.actions[3].append(did_riichi)

                logging.debug(f'{round["round"]} | player={player_idx}, turn={turn}, did_riichi={did_riichi}')

    def get_discard_model_data(self, round_state, round, event):
        if event["type"] == "Discard":
            player_idx = event["player"]
            if round_state._others_riichi_status[player_idx] == 1:
                return

            self.states[4].append(round_state._player_states[player_idx])
            self.actions[4].append(event["tile"])

            logging.debug(f'{round["round"]} | player{player_idx}, turn={round_state._player_states[player_idx]._turn + 1}, discard={event["tile"]}')

    def update_player_scores(self, round, scores, deltas):
        new_scores = [a + b for a, b in zip(scores, deltas)]

        if len(round["reaches"]) == 0:
            return new_scores

        for player_idx in round["reaches"]:
            new_scores[player_idx] -= 10

        return new_scores

    def export_to_csv(self):
        for i, (states, actions) in enumerate(zip(self.states, self.actions)):
            actions = list(map(lambda a: int(a) if a in [0,1,2,3] else a[0:2], actions)) 

            match i:
                case 0: csv_file_name = "../data/model_data/chii_data.csv"
                case 1: csv_file_name = "../data/model_data/pon_data.csv"
                case 2: csv_file_name = "../data/model_data/kan_data.csv"
                case 3: csv_file_name = "../data/model_data/riichi_data.csv"
                case 4: csv_file_name = "../data/model_data/discard_data.csv"

            # with open(csv_file_name, mode="a+") as csv_file:
            with open(csv_file_name, mode="w") as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                for state, action in zip(states, actions):
                    row = [f for f in state.to_features_list()]
                    row.append(action)
                    writer.writerow(row)

        logging.debug(time.time())

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

        self._p0_state = PlayerState(0, self._round, self._player_scores, self._discarded_tiles, self._open_tiles, self._dora_indicators, self.get_player_wind(0, self._dealer), self._others_riichi_status)
        self._p1_state = PlayerState(1, self._round, self._player_scores, self._discarded_tiles, self._open_tiles, self._dora_indicators, self.get_player_wind(1, self._dealer), self._others_riichi_status)
        self._p2_state = PlayerState(2, self._round, self._player_scores, self._discarded_tiles, self._open_tiles, self._dora_indicators, self.get_player_wind(2, self._dealer), self._others_riichi_status)
        self._p3_state = PlayerState(3, self._round, self._player_scores, self._discarded_tiles, self._open_tiles, self._dora_indicators, self.get_player_wind(3, self._dealer), self._others_riichi_status)

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

class PlayerState:
    def __init__(self, player_idx, round, player_scores, discarded_tiles, open_tiles, dora_indicators, wind, riichi_status):
        self._player_idx = player_idx 
        self._discarded_tiles = discarded_tiles
        self._open_tiles = open_tiles
        self._round_info = round["round"] # Tuple of string (name), int (honba count), int (leftover riichi sticks)
        self._turn = 0
        self._riichi_status = riichi_status
        self._is_open_hand = False

        # features
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
        # self._others_riichi_status = self._riichi_status[:self._player_idx] + self._riichi_status[self._player_idx + 1:]

        self.update_aka_dora_count_in_hand()

    def to_features_list(self):
        return [
            self._private_tiles,
            self._private_discarded_tiles,
            self._others_discarded_tiles,
            self._private_open_tiles,
            self._others_open_tiles,
            self._dora_indicators,
            self._round_name,
            self._player_scores,
            self._self_wind,
            self._aka_doras_in_hand,
            self._riichi_status[:self._player_idx] + self._riichi_status[self._player_idx+1:]
        ]

    def update_aka_dora_count_in_hand(self):
        aka_doras = ["5m0", "5s0", "5p0"]

        aka_dora_count_private = sum([1 if (i in aka_doras) else 0 for i in self._private_tiles])
        aka_dora_count_open = sum([1 if (i in aka_doras) else 0 for i in self._private_open_tiles])

        self._aka_doras_in_hand = aka_dora_count_private + aka_dora_count_open

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

    def can_chii(self, tile) -> Boolean:
        number = tile[0:1]
        suit = tile[1:2]

        if suit == "w" or suit == "d":
            return False

        transformed_hand = self.transform_hand(self._private_tiles)

        offset = 0
        if suit == "p": offset = 9
        if suit == "s": offset = 18

        for combination in chii_combinations[number]:
            if transformed_hand[offset + int(combination[0])-1] > 0 \
                and transformed_hand[offset + int(combination[1])-1] > 0:
                return True

        return False 

    def can_pon(self, tile) -> Boolean:
        transformed_hand = self.transform_hand(self._private_tiles)

        tile = tile[0:2]
        tile_idx = tiles.index(tile)

        if transformed_hand[tile_idx] >= 2:
            return True

        return False

    def can_chakan(self) -> Boolean:
        transformed_hand = self.transform_hand(self._private_tiles)

        for tile_cnt in transformed_hand:
            if tile_cnt == 4:
                return True

        return False
        

    def can_kan(self, tile) -> Boolean:
        transformed_hand = self.transform_hand(self._private_tiles)

        tile = tile[0:2]
        tile_idx = tiles.index(tile)

        if transformed_hand[tile_idx] >= 3:
            return True

        return False

    def can_riichi(self) -> Boolean:
        if self._player_scores[self._player_idx] < 10:
            return False

        if self._riichi_status[self._player_idx] == 1:
            return False

        if self._is_open_hand == True:
            return False

        transformed_hand = self.transform_hand(deepcopy(self._private_tiles))

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

        counts = [0, 0, 0, 0]
        combinations = [m_combinations, p_combinations, s_combinations, w_combinations, d_combinations]
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

        # TODO: check for kokushi musou (maybe, might make the model worse?)

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
            meld = tiles[idx:idx+3]

            if 0 in meld:
                continue
                
            count = sum(meld) // 3
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
            meld = tiles[idx:idx+3]

            if sum(meld) == 2:
                taatsu_count += 1
                if tiles[idx] == 1:
                    tiles[idx] -= 1

                if tiles[idx + 1] == 1:
                    tiles[idx + 1] -= 1

                if tiles[idx + 2] == 1:
                    tiles[idx + 2] -= 1

        # check isolated 
        isolated_count = sum(tiles)

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
        isolated_count = sum(tiles)

        return (mentsu_count, pair_count, 0, isolated_count)
        
    def transform_hand(self, hand):
        new_hand = [0] * 34
        mapped_hand = list(map(lambda x: x[0:2], hand))

        for tile in mapped_hand:
            idx = tiles.index(tile)
            new_hand[idx] += 1

        return new_hand

    def __str__(self):
        return (
            "---------------------------------------\n"
            f"_private_tiles: {sort_hand(self._private_tiles)}\n"
            f"_private_discarded_tiles: {self._private_discarded_tiles}\n"
            f"_others_discarded_tiles: {self._others_discarded_tiles}\n"
            f"_others_open_tiles: {self._others_open_tiles}\n"
            f"_dora_indicators: {self._dora_indicators}\n"
            f"_round_name: {self._round_name}\n"
            f"_player_scores: {self._player_scores}\n"
            f"_self_wind: {self._self_wind}\n"
            f"_aka_doras_in_hand: {self._aka_doras_in_hand}\n"
            f"_riichi_status: {self._riichi_status}\n"
            "---------------------------------------\n"
        )

def sort_hand(tiles):
    return sorted(tiles, key = lambda x: (x[1], x[0]))

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

chii_combinations = {
    "1": ["23"],
    "2": ["13", "34"],
    "3": ["12", "24", "45"],
    "4": ["23", "35", "56"],
    "5": ["34", "46", "67"],
    "6": ["45", "57", "78"],
    "7": ["56", "68", "89"],
    "8": ["67", "79"],
    "9": ["78"],
}

def main(logs_dir):
    game = Game('DEFAULT')

    # todo use glob to check all files in subfolders at ../tenhou-logs/xml-logs/
    log_paths = [logs_dir + f for f in os.listdir(logs_dir)]
    for log_path in log_paths:
        logging.info(log_path)

        game.decode(open(log_path))
        game_data = game.asdata()

        parser = GameLogParser(game_data) 

        parser.get_nn_input()

        # break
        

if __name__ == '__main__':
    logging.basicConfig(level=LOGGING_LEVEL)

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--folder", help="path to folder containing .xml-logs")

    args = parser.parse_args()

    main(args.folder)
