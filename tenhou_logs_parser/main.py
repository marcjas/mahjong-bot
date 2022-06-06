import sys
import logging
import argparse
import pickle
from tqdm import tqdm
from copy import deepcopy
import re
from typing import Dict
import glob 
import torch

from tenhou_decoder import Game

LOGGING_LEVEL = logging.INFO

class GameLogParser:
    def __init__(self, data: Dict):
        self._data = data
        self._players = data["players"]
        self._rounds = data["rounds"]

    def get_nn_input(self, get_chii_model_input=True, get_pon_model_input=True, get_kan_model_input=True, get_riichi_model_input=True, get_discard_model_input=True):
        player_scores = [250, 250, 250, 250]
        for round in self._rounds:
            round_state = RoundState(round, player_scores)
            for event_idx, event in enumerate(round["events"]):
                if get_discard_model_input:
                    self.get_discard_model_data(round_state, round, event)

                round_state.update_state(event)

                if get_chii_model_input:
                    self.get_chii_model_data(round_state, round, event, event_idx)

                if get_pon_model_input:
                    self.get_pon_model_data(round_state, round, event, event_idx)

                if get_kan_model_input:
                    self.get_kan_model_data(round_state, round, event, event_idx)

                if get_riichi_model_input:
                    self.get_riichi_model_data(round_state, round, event)

            player_scores = self.update_player_scores(round, player_scores, round["deltas"])

            logging.debug(f'{round["round"]} | scores={player_scores}')

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

                batch_states[0].append(deepcopy(round_state._player_states[player_idx]))
                batch_actions[0].append(int(did_chii))

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

                    batch_states[1].append(deepcopy(round_state._player_states[player_idx]))
                    batch_actions[1].append(int(did_pon))

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

                batch_states[2].append(deepcopy(round_state._player_states[player_idx]))
                batch_actions[2].append(int(did_chakan))

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

                    batch_states[2].append(deepcopy(round_state._player_states[player_idx]))
                    batch_actions[2].append(int(did_kan))

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
                
                batch_states[3].append(deepcopy(round_state._player_states[player_idx]))
                batch_actions[3].append(int(did_riichi))

                logging.debug(f'{round["round"]} | player={player_idx}, turn={turn}, did_riichi={did_riichi}')

    def get_discard_model_data(self, round_state, round, event):
        if event["type"] == "Discard":
            player_idx = event["player"]
            if round_state._others_riichi_status[player_idx] == 1:
                return

            batch_states[4].append(deepcopy(round_state._player_states[player_idx]))
            batch_actions[4].append(event["tile"])

            logging.debug(f'{round["round"]} | player{player_idx}, turn={round_state._player_states[player_idx]._turn + 1}, discard={event["tile"]}')

    def update_player_scores(self, round, scores, deltas):
        new_scores = [a + b for a, b in zip(scores, deltas)]

        if len(round["reaches"]) == 0:
            return new_scores

        for player_idx in round["reaches"]:
            new_scores[player_idx] -= 10

        return new_scores

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
        winds = ["東", "南", "西", "北"]
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

        # right, opposite, left
        self._others_order = [((self._player_idx+i+1) % 4) for i in range(3)]

        # features
        self._private_tiles = round["hands"][self._player_idx]
        self._private_discarded_tiles = self._discarded_tiles[self._player_idx]
        self._others_discarded_tiles = [self._discarded_tiles[i] for i in self._others_order]
        self._private_open_tiles = self._open_tiles[self._player_idx] 
        self._others_open_tiles = [self._open_tiles[i] for i in self._others_order]
        self._dora_indicators = dora_indicators
        self._round_name = self._round_info[0]
        self._player_scores = [player_scores[self._player_idx]] + [player_scores[i] for i in self._others_order]
        self._self_wind = wind
        self._aka_doras_in_hand = 0

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
            [self._riichi_status[i] for i in self._others_order]
        ]


    def to_processed_features_list(self):
        """
        private_tiles:                  34 x 4
        private_discarded_tiles:        34 x 30
        others_discarded_tiles:         34 x 30 x 3
        private_open_tiles:             34 x 4
        others_open_tiles:              34 x 4 x 3
        dora_indicators:                34 x 4
        round_name:                     12
        player_scores:                  4
        self_wind:                      4
        aka_doras_in_hand:              4
        riichi_status:                  3
        """

        # self._private_tiles
        private_tiles = [0] * 34 * 4
        mapped_private_tiles = list(map(lambda x: x[0:2], self._private_tiles))
        for tile in mapped_private_tiles:
            idx = tiles.index(tile)
            for i in range(4):
                if private_tiles[idx + i * 34] == 0:
                    private_tiles[idx + i * 34] = 1
                    break

        # self._private_discarded_tiles
        private_discarded_tiles = [0] * 34 * 30
        mapped_private_discarded_tiles = list(map(lambda x: x[0:2], self._private_discarded_tiles))
        for i, tile in enumerate(mapped_private_discarded_tiles):
            idx = tiles.index(tile)
            private_discarded_tiles[idx + (i * 34)] = 1

        # self._others_discarded_tiles
        private_others_discarded_tiles = [0] * 34 * 30 * 3
        for player_i, discarded_tiles in enumerate(self._others_discarded_tiles):
            mapped_player_discarded_tiles = list(map(lambda x: x[0:2], discarded_tiles)) 
            for i, tile in enumerate(mapped_player_discarded_tiles):
                idx = tiles.index(tile)
                private_others_discarded_tiles[idx + (i * 34) + (player_i * 34 * 30)] = 1

        # self._private_open_tiles
        private_open_tiles = [0] * 34 * 4
        for i, meld in enumerate(self._private_open_tiles):
            mapped_meld = list(map(lambda x: x[0:2], meld))
            for tile in mapped_meld:
                idx = tiles.index(tile)
                for j in range(4):
                    if private_open_tiles[idx + (j * 34)] == 0:
                        private_open_tiles[idx + (j * 34)] = 1
                        break

        # self._others_open_tiles
        others_open_tiles = [0] * 34 * 4 * 3
        for player_i, player_melds in enumerate(self._others_open_tiles):
            for i, meld in enumerate(player_melds):
                mapped_meld = list(map(lambda x: x[0:2], meld))
                for tile in mapped_meld:
                    idx = tiles.index(tile)
                    for j in range(4):
                        if others_open_tiles[idx + (j * 34 + (player_i * 34 * 4))] == 0:
                            others_open_tiles[idx + (j * 34 + (player_i * 34 * 4))] = 1
                            break

        # self._dora_indicators
        dora_indicators = [0] * 34 * 4
        mapped_dora_indicators = list(map(lambda x: x[0:2], self._dora_indicators))
        for tile in mapped_dora_indicators:
            idx = tiles.index(tile)
            for i in range(4):
                if dora_indicators[idx + i * 34] == 0:
                    dora_indicators[idx + i * 34] = 1
                    break

        # self._round_name
        round_name = [0] * 12
        winds = ["東", "南", "西", "北"]
        round_wind = self._round_name[0]
        round_number = int(self._round_name[1])
        round_name[winds.index(round_wind) + round_number]

        # self._player_scores
        player_scores = [0] * 4
        for i, player_score in enumerate(self._player_scores):
            if player_score >= 1000:
                player_scores[i] = 1
            elif player_score <= 0:
                player_scores[i] = 0
            else:
                player_scores[i] = player_score / 1000

        # self._self_wind
        self_wind = [0] * 4
        winds = ["東", "南", "西", "北"]
        self_wind[winds.index(self._self_wind)] = 1

        # self._aka_doras_in_hand
        aka_doras_in_hand = [0] * 4
        aka_doras_in_hand[self._aka_doras_in_hand] = 1

        # self._riichi_status
        others_riichi_status = [self._riichi_status[i] for i in self._others_order]

        return private_tiles + \
               private_discarded_tiles + \
               private_others_discarded_tiles + \
               private_open_tiles + \
               others_open_tiles + \
               dora_indicators + \
               round_name + \
               player_scores + \
               self_wind + \
               aka_doras_in_hand + \
               others_riichi_status

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

    def can_chii(self, tile):
        number = tile[0:1]
        suit = tile[1:2]

        if suit == "w" or suit == "d":
            return False

        transformed_hand = transform_hand(self._private_tiles)

        offset = 0
        if suit == "p": offset = 9
        if suit == "s": offset = 18

        for combination in chii_combinations[number]:
            if transformed_hand[offset + int(combination[0])-1] > 0 \
                and transformed_hand[offset + int(combination[1])-1] > 0:
                return True

        return False 

    def can_pon(self, tile):
        transformed_hand = transform_hand(self._private_tiles)

        tile = tile[0:2]
        tile_idx = tiles.index(tile)

        if transformed_hand[tile_idx] >= 2:
            return True

        return False

    def can_chakan(self):
        transformed_hand = transform_hand(self._private_tiles)

        for tile_cnt in transformed_hand:
            if tile_cnt == 4:
                return True

        return False
        

    def can_kan(self, tile):
        transformed_hand = transform_hand(self._private_tiles)

        tile = tile[0:2]
        tile_idx = tiles.index(tile)

        if transformed_hand[tile_idx] >= 3:
            return True

        return False

    def can_riichi(self):
        if self._player_scores[self._player_idx] < 10:
            return False

        if self._riichi_status[self._player_idx] == 1:
            return False

        if self._is_open_hand == True:
            return False

        transformed_hand = transform_hand(deepcopy(self._private_tiles))

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
        
    

    def __str__(self):
        return (
            "---------------------------------------\n"
            f"_private_tiles: {sort_hand(self._private_tiles)}\n"
            f"_private_discarded_tiles: {self._private_discarded_tiles}\n"
            f"_others_discarded_tiles: {self._others_discarded_tiles}\n"
            f"_private_open_tiles: {self._private_open_tiles}\n"
            f"_others_open_tiles: {self._others_open_tiles}\n"
            f"_dora_indicators: {self._dora_indicators}\n"
            f"_round_name: {self._round_name}\n"
            f"_player_scores: {self._player_scores}\n"
            f"_self_wind: {self._self_wind}\n"
            f"_aka_doras_in_hand: {self._aka_doras_in_hand}\n"
            f"_riichi_status: {self._riichi_status}\n"
            "---------------------------------------\n"
        )

def transform_hand(hand):
    new_hand = [0] * 34
    mapped_hand = list(map(lambda x: x[0:2], hand))

    for tile in mapped_hand:
        idx = tiles.index(tile)
        new_hand[idx] += 1

    return new_hand

def sort_hand(tiles):
    return sorted(tiles, key = lambda x: (x[1], x[0]))

def decode_tile(tile_code):
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

batch_size = 100 # game
batch_states = [[], [], [], [], []]
batch_actions = [[], [], [], [], []]

MAX_CHII_DATA = 0
MAX_PON_DATA = 0 
MAX_KAN_DATA = 0
MAX_RIICHI_DATA = 0
MAX_DISCARD_DATA = 200000 
MAX_DATA = [MAX_CHII_DATA, MAX_PON_DATA, MAX_KAN_DATA, MAX_RIICHI_DATA, MAX_DISCARD_DATA]

def get_metadata():
    try:
        metadata = pickle.load(open("metadata.pickle", "rb"))
    except (OSError, IOError) as error:
        metadata = [0, 0, 0, 0, 0]
        pickle.dump(metadata, open("metadata.pickle", "wb"))

    return metadata

def update_metadata(new_metadata):
    pickle.dump(new_metadata, open("metadata.pickle", "wb"))

def export():
    metadata = get_metadata()
    for i, (states, actions) in enumerate(zip(batch_states, batch_actions)):  
        match i:
            case 0: 
                memmap_file_name = "../data/model_data/chii_data.dat"
                vector_length = 4923 + 1
            case 1: 
                memmap_file_name = "../data/model_data/pon_data.dat"
                vector_length = 4923 + 1
            case 2: 
                memmap_file_name = "../data/model_data/kan_data.dat"
                vector_length = 4923 + 1
            case 3: 
                memmap_file_name = "../data/model_data/riichi_data.dat"
                vector_length = 4923 + 1
            case 4: 
                memmap_file_name = "../data/model_data/discard_data.dat"
                vector_length = 4923 + 1
            case _: sys.exit(-1)

        # shared=True allows us to save the tensor to disk as we perform in place modifications to it
        storage = torch.FloatStorage.from_file(memmap_file_name, shared=True, size=MAX_DATA[i] * vector_length)
        memmap = torch.FloatTensor(storage).reshape(MAX_DATA[i], vector_length)

        for state, action in zip(states, actions):
            if metadata[i] >= MAX_DATA[i]:
                break
            if i == 4:
                memmap[metadata[i]] = torch.FloatTensor(state.to_processed_features_list() + [tiles.index(action[0:2])])
            else:
                memmap[metadata[i]] = torch.FloatTensor(state.to_processed_features_list() + [action])
            metadata[i] += 1

    update_metadata(metadata)
    print(f".dat lengths: {metadata}")
    print("Finished exporting")


def main(logs_dir):
    game = Game('DEFAULT')
    
    metadata = get_metadata()
    print(f".dat lengths: {metadata}")
    if sum([1 if metadata[x] >= MAX_DATA[x] else 0 for x in range(5)]) == 5:
        print("Max data...")
        sys.exit(-1)

    log_paths = glob.glob(f"{logs_dir}/**/*.xml", recursive=True)
    if len(log_paths) == 0:
        print("Found no logs...")
        sys.exit(-1)

    for i in range(0, len(log_paths), batch_size):
        batch = log_paths[i:i+batch_size] 

        metadata = get_metadata()
        if sum([1 if metadata[x] >= MAX_DATA[x] else 0 for x in range(5)]) == 5:
            print("Finished collecting all data...")
            break

        for log_path in tqdm(batch):
            game.decode(open(log_path))
            game_data = game.asdata()

            parser = GameLogParser(game_data)

            get_data = [False if metadata[x] >= MAX_DATA[x] else True for x in range(5)]
            parser.get_nn_input(get_data[0], get_data[1], get_data[2], get_data[3], get_data[4])

        export()

        global batch_states 
        global batch_actions 

        batch_states = [[], [], [], [], []]
        batch_actions = [[], [], [], [], []]

if __name__ == '__main__':
    logging.basicConfig(level=LOGGING_LEVEL)

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--folder", help="path to folder containing .xml-logs")

    args = parser.parse_args()

    main(args.folder)
