import sys
import logging
import argparse
from tqdm import tqdm
from copy import deepcopy
import re
import glob 
import torch
import os

sys.path.append("utils")
import mahjong_utils
from constants import TILES, CHII_COMBINATIONS, WINDS
from tenhou_decoder import Game

LOGGING_LEVEL = logging.INFO

# config
BATCH_SIZE = 100
EXPORT_DATA = True
MAX_DATA = {
    "chii": 0,
    "pon": 0,
    "kan": 0,
    "riichi": 0,
    "discard": 5000,
}
DISCARD_FEATURES = sum([
    34 * 4,      # player tiles
    34 * 27 * 4, # discarded tiles
    34 * 4 * 4,  # open tiles
    34 * 4,      # dora indicators
    34,          # shanten increase per discard
    34 * 47,     # improvement tiles per discard (buckets, start at range 1, increase by 1 every 10 buckets)
    12,          # round name
    4 * 20,      # player scores (buckets from 0 to 1000, range 50)
    4,           # player wind
    4,           # aka doras in hand
    3])          # others' riichi status

class Data:
    _data = {
        "chii": { "states": [], "actions": [], "writes": 0 },
        "pon": { "states": [], "actions": [], "writes": 0 },
        "kan": { "states": [], "actions": [], "writes": 0 },
        "riichi": { "states": [], "actions": [], "writes": 0 },
        "discard": { "states": [], "actions": [], "writes": 0 },
    }

    def __getitem__(self, item):
        return self._data[item]

    def __iter__(self):
        return iter(self._data)

    def can_write(self):
        return self["chii"]["writes"] < MAX_DATA["chii"], \
            self["pon"]["writes"] < MAX_DATA["pon"], \
            self["kan"]["writes"] < MAX_DATA["kan"], \
            self["riichi"]["writes"] < MAX_DATA["kan"], \
            self["discard"]["writes"] < MAX_DATA["discard"]

    def is_full(self):
        return all((self["chii"]["writes"] >= MAX_DATA["chii"],
            self["pon"]["writes"] >= MAX_DATA["pon"],
            self["kan"]["writes"] >= MAX_DATA["kan"],
            self["riichi"]["writes"] >= MAX_DATA["kan"],
            self["discard"]["writes"] >= MAX_DATA["discard"]))

    def export(self):
        for key in self:
            memmap_fn = f"data/model_data/{key}_data.dat"
            vector_len = DISCARD_FEATURES + 1
            
            storage = torch.FloatStorage.from_file(memmap_fn, shared=True, size=MAX_DATA[key] * vector_len)
            memmap = torch.FloatTensor(storage).reshape(MAX_DATA[key], vector_len)

            data = self[key]
            for i, (state, action) in tqdm(enumerate(zip(data["states"], data["actions"]))):
                if data["writes"] >= MAX_DATA[key]: 
                    break
                if key == "discard":
                    memmap[data["writes"]] = torch.FloatTensor(state.to_feature_list() + [TILES.index(action[0:2])])
                else:
                    memmap[data["writes"]] = torch.FloatTensor(state.to_feature_list() + [action])
                data["writes"] += 1

            data["states"] = []
            data["actions"] = []

        print(f'Chii: {self["chii"]["writes"]}, pon: {self["pon"]["writes"]}, kan: {self["kan"]["writes"]}, riichi: {self["riichi"]["writes"]}, discard: {self["discard"]["writes"]}')

class GameLogParser:
    def __init__(self, data: Data, game_data):
        self._data = data
        self._players = game_data["players"]
        self._rounds = game_data["rounds"]

    def get_data(self):
        get_chii_data, get_pon_data, get_kan_data, get_riichi_data, get_discard_data = self._data.can_write()
        player_scores = [250, 250, 250, 250]
        for round in self._rounds:
            round_state = RoundState(round, player_scores)
            for event_idx, event in enumerate(round["events"]):
                if get_discard_data:
                    self.get_discard_model_data(round_state, round, event)

                round_state.update_state(event)

                if get_chii_data:
                    self.get_chii_model_data(round_state, round, event, event_idx)

                if get_pon_data:
                    self.get_pon_model_data(round_state, round, event, event_idx)

                if get_kan_data:
                    self.get_kan_model_data(round_state, round, event, event_idx)

                if get_riichi_data:
                    self.get_riichi_model_data(round_state, round, event)

            player_scores = self.update_player_scores(round, player_scores, round["deltas"])

            logging.debug(f'{round["round"]} | scores={player_scores}')

    def get_chii_model_data(self, round_state, round, event, event_idx):
        if event["type"] == "Discard":
            discard_player_idx = event["player"]
            player_idx = ((discard_player_idx + 1) + 4) % 4
            
            if round_state._riichi_status[0][player_idx] == 1:
                return 

            if round_state._player_states[player_idx].can_chii(event["tile"]):
                did_chii = False
                next_event_idx = event_idx + 1

                try:
                    next_event = round["events"][next_event_idx]
                except IndexError:
                    return 

                if next_event["type"] == "Call" and next_event["meld"]["type"] == "chi" and next_event["player"] == player_idx:
                    did_chii = True

                self._data["chii"]["states"].append(deepcopy(round_state._player_states[player_idx]))
                self._data["chii"]["actions"].append(int(did_chii))

                logging.debug(f'{round["round"]} | player={player_idx}, turn={round_state._player_states[player_idx]._turn}, did_chii={did_chii}')

    def get_pon_model_data(self, round_state, round, event, event_idx):
        if event["type"] == "Discard":
            discard_player_idx = event["player"]
            for player_idx in [x for x in range(4) if x != discard_player_idx]:
                if round_state._riichi_status[0][player_idx] == 1:
                    continue
                
                if round_state._player_states[player_idx].can_pon(event["tile"]):
                    did_pon = False
                    next_event_idx = event_idx + 1

                    try:
                        next_event = round["events"][next_event_idx]
                    except IndexError:
                        break 

                    if next_event["type"] == "Call" and next_event["meld"]["type"] == "pon" and next_event["player"] == player_idx:
                        did_pon = True

                    self._data["pon"]["states"].append(deepcopy(round_state._player_states[player_idx]))
                    self._data["pon"]["actions"].append(int(did_pon))

                    logging.debug(f'{round["round"]} | player={player_idx}, turn={round_state._player_states[player_idx]._turn}, did_pon={did_pon}')

    def get_kan_model_data(self, round_state, round, event, event_idx):
        if event["type"] == "Draw":
            player_idx = event["player"]
            if round_state._player_states[player_idx].can_chakan():
                did_chakan = False
                next_event_idx = event_idx + 1

                try:
                    next_event = round["events"][next_event_idx]
                except IndexError:
                    return 

                if next_event["type"] == "Call" and next_event["meld"]["type"] == "chakan" and next_event["player"] == player_idx:
                    did_chakan = True

                self._data["kan"]["states"].append(deepcopy(round_state._player_states[player_idx]))
                self._data["kan"]["actions"].append(int(did_chakan))

                logging.debug(f'{round["round"]} | player={player_idx}, turn={round_state._player_states[player_idx]._turn}, did_chakan={did_chakan}')

        if event["type"] == "Discard":
            discard_player_idx = event["player"]
            for player_idx in [x for x in range(4) if x != discard_player_idx]:
                if round_state._riichi_status[0][player_idx] == 1:
                    continue

                if round_state._player_states[player_idx].can_kan(event["tile"]):
                    did_kan = False
                    next_event_idx = event_idx + 1

                    try:
                        next_event = round["events"][next_event_idx]
                    except IndexError:
                        return 

                    if next_event["type"] == "Call" and next_event["meld"]["type"] == "kan" and next_event["player"] == player_idx:
                        did_kan = True

                    self._data["kan"]["states"].append(deepcopy(round_state._player_states[player_idx]))
                    self._data["kan"]["actions"].append(int(did_kan))

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
                
                self._data["riichi"]["states"].append(deepcopy(round_state._player_states[player_idx]))
                self._data["riichi"]["actions"].append(int(did_riichi))

                logging.debug(f'{round["round"]} | player={player_idx}, turn={turn}, did_riichi={did_riichi}')

    def get_discard_model_data(self, round_state, round, event):
        if event["type"] == "Discard":
            player_idx = event["player"]
            if round_state._riichi_status[0][player_idx] == 1:
                return

            self._data["discard"]["states"].append(deepcopy(round_state._player_states[player_idx]))
            self._data["discard"]["actions"].append(event["tile"])

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
        self._riichi_status = [[0, 0, 0, 0]]
        self._riichis = self._round["reaches"]
        self._riichi_turns = self._round["reach_turns"]
        self._riichis_dict = { self._riichis[i]: self._riichi_turns[i] for i in range(len(self._riichis)) }

        self._p0_state = PlayerState(0, self._round, self._player_scores, self._discarded_tiles, self._open_tiles, self._dora_indicators, self.get_player_wind(0, self._dealer), self._riichi_status)
        self._p1_state = PlayerState(1, self._round, self._player_scores, self._discarded_tiles, self._open_tiles, self._dora_indicators, self.get_player_wind(1, self._dealer), self._riichi_status)
        self._p2_state = PlayerState(2, self._round, self._player_scores, self._discarded_tiles, self._open_tiles, self._dora_indicators, self.get_player_wind(2, self._dealer), self._riichi_status)
        self._p3_state = PlayerState(3, self._round, self._player_scores, self._discarded_tiles, self._open_tiles, self._dora_indicators, self.get_player_wind(3, self._dealer), self._riichi_status)

        self._player_states = [self._p0_state, self._p1_state, self._p2_state, self._p3_state]

    def get_player_wind(self, player_idx, dealer_idx) -> str:
        return WINDS[(player_idx - dealer_idx) % 4]

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
                    self._riichi_status[0][player_idx] = 1

            case "Draw":
                player_idx = event["player"]
                self._player_states[player_idx].draw(event["tile"])

            case "Call":
                player_idx = event["player"]
                self._player_states[player_idx].call(event["meld"])

        for player in self._player_states:
            player.update_aka_dora_count_in_hand()
    

class PlayerState:
    def __init__(self, player_idx, round, player_scores, discarded_tiles, open_tiles, dora_indicators, wind, riichi_status):
        self._player_idx = player_idx 
        self._discarded_tiles = discarded_tiles
        self._open_tiles = open_tiles
        self._round_info = round["round"] # Tuple of string (name), int (honba count), int (leftover riichi sticks)
        self._turn = 0
        self._riichi_status = riichi_status[0]
        self._is_open_hand = False

        # right, opposite, left
        self._others_order = [((self._player_idx+i+1) % 4) for i in range(3)]

        # features
        self._player_tiles = round["hands"][self._player_idx]
        self._player_discarded_tiles = self._discarded_tiles[self._player_idx]
        self._others_discarded_tiles = [self._discarded_tiles[i] for i in self._others_order]
        self._player_open_tiles = self._open_tiles[self._player_idx] 
        self._others_open_tiles = [self._open_tiles[i] for i in self._others_order]
        self._dora_indicators = dora_indicators
        self._round_name = self._round_info[0]
        self._player_scores = [player_scores[self._player_idx]] + [player_scores[i] for i in self._others_order]
        self._player_wind = wind
        self._aka_doras_in_hand = 0
        self._others_riichi_status = [self._riichi_status[i] for i in self._others_order]

        self.update_aka_dora_count_in_hand()

    def to_feature_list(self):
        """
        player_tiles:                  34 x 4
        player_discarded_tiles:        34 x 27
        others_discarded_tiles:        34 x 27 x 3
        player_open_tiles:             34 x 4
        others_open_tiles:             34 x 4 x 3
        dora_indicators:               34 x 4
        better_shanten                 34
        improvement_tiles              34 x 47
        round_name:                    12
        player_scores:                 4 x 20
        player_wind:                   4
        aka_doras_in_hand:             4
        others_riichi_status:          3
                                       = 6147
        """

        # self._player_tiles
        player_tiles = [0] * 34 * 4
        mapped_player_tiles = list(map(lambda x: x[0:2], self._player_tiles))
        for tile in mapped_player_tiles:
            idx = TILES.index(tile)
            for i in range(4):
                if player_tiles[idx + i * 34] == 0:
                    player_tiles[idx + i * 34] = 1
                    break

        # self._player_discarded_tiles
        player_discarded_tiles = [0] * 34 * 27
        mapped_plaer_discarded_tiles = list(map(lambda x: x[0:2], self._player_discarded_tiles))
        for i, tile in enumerate(mapped_plaer_discarded_tiles):
            idx = TILES.index(tile)
            player_discarded_tiles[idx + (i * 34)] = 1

        # self._others_discarded_tiles
        others_discarded_tiles = [0] * 34 * 27 * 3
        for player_i, discarded_tiles in enumerate(self._others_discarded_tiles):
            mapped_player_discarded_tiles = list(map(lambda x: x[0:2], discarded_tiles)) 
            for i, tile in enumerate(mapped_player_discarded_tiles):
                idx = TILES.index(tile)
                others_discarded_tiles[idx + (i * 34) + (player_i * 34 * 27)] = 1

        # self._player_open_tiles
        player_open_tiles = [0] * 34 * 4
        for i, meld in enumerate(self._player_open_tiles):
            mapped_meld = list(map(lambda x: x[0:2], meld))
            for tile in mapped_meld:
                idx = TILES.index(tile)
                for j in range(4):
                    if player_open_tiles[idx + (j * 34)] == 0:
                        player_open_tiles[idx + (j * 34)] = 1
                        break

        # self._others_open_tiles
        others_open_tiles = [0] * 34 * 4 * 3
        for player_i, player_melds in enumerate(self._others_open_tiles):
            for i, meld in enumerate(player_melds):
                mapped_meld = list(map(lambda x: x[0:2], meld))
                for tile in mapped_meld:
                    idx = TILES.index(tile)
                    for j in range(4):
                        if others_open_tiles[idx + (j * 34 + (player_i * 34 * 4))] == 0:
                            others_open_tiles[idx + (j * 34 + (player_i * 34 * 4))] = 1
                            break

        # self._dora_indicators
        dora_indicators = [0] * 34 * 4
        mapped_dora_indicators = list(map(lambda x: x[0:2], self._dora_indicators))
        for tile in mapped_dora_indicators:
            idx = TILES.index(tile)
            for i in range(4):
                if dora_indicators[idx + i * 34] == 0:
                    dora_indicators[idx + i * 34] = 1
                    break

        # better shanten
        better_shanten = [0] * 34
        potential_shanten = mahjong_utils.get_potential_shanten(self._player_tiles)
        best_shanten = min(potential_shanten)
        for i, tile in enumerate(self._player_tiles):
            if potential_shanten[i] == best_shanten:
                idx = TILES.index(tile[:2])
                better_shanten[idx] = 1

        def get_bucket(x):
            bucket = 0
            i = 1
            while i < 136:
                if x <= i:
                    return bucket
                bucket += 1
                i += (1 + bucket // 10)
            return bucket - 1

        improvement_tiles = [0] * 34 * 47
        #tile_counts = mahjong_utils.get_improvement_tiles(self._player_tiles)
        #for i, tile in enumerate(self._player_tiles):
        #    bucket = get_bucket(tile_counts[i])
        #    idx = TILES.index(tile[:2])
        #    improvement_tiles[bucket * 34 + idx]

        # self._round_name
        round_name = [0] * 12
        round_wind = self._round_name[0]
        round_number = int(self._round_name[1])
        round_name[WINDS.index(round_wind) + round_number]

        def get_bucket(x):
            if x >= 1000:
                return 19
            elif x <= 0:
                return 0
            return x // 50

        # self._player_scores
        player_scores = [0] * 4 * 20
        for i, player_score in enumerate(self._player_scores):
            bucket = get_bucket(player_score)
            player_scores[bucket * 4 + i] = 1

        # self._player_wind
        player_wind = [0] * 4
        player_wind[WINDS.index(self._player_wind)] = 1

        # self._aka_doras_in_hand
        aka_doras_in_hand = [0] * 4
        aka_doras_in_hand[self._aka_doras_in_hand] = 1

        return player_tiles + \
               player_discarded_tiles + \
               others_discarded_tiles + \
               player_open_tiles + \
               others_open_tiles + \
               dora_indicators + \
               better_shanten + \
               improvement_tiles + \
               round_name + \
               player_scores + \
               player_wind + \
               aka_doras_in_hand + \
               self._others_riichi_status

    def update_aka_dora_count_in_hand(self):
        aka_doras = ["5m0", "5s0", "5p0"]

        aka_dora_count_private = sum([1 if (i in aka_doras) else 0 for i in self._player_tiles])
        aka_dora_count_open = sum([1 if (i in aka_doras) else 0 for i in self._player_open_tiles])

        self._aka_doras_in_hand = aka_dora_count_private + aka_dora_count_open

    def discard(self, tile):
        self._player_tiles.remove(tile)
        self._discarded_tiles[self._player_idx].append(tile)
        self._turn += 1

    def draw(self, tile):
        self._player_tiles.append(tile)

    def call(self, meld):
        tiles = meld["tiles"]

        if meld["type"] == "kan":
            self._is_open_hand = True

        for tile in tiles:
            if tile in self._player_tiles:
                self._player_tiles.remove(tile)

        self._open_tiles[self._player_idx].append(tiles)

    def can_chii(self, tile) -> bool:
        number = tile[0:1]
        suit = tile[1:2]

        if suit == "w" or suit == "d":
            return False

        transformed_hand = transform_hand(self._player_tiles)

        offset = 0
        if suit == "p": offset = 9
        if suit == "s": offset = 18

        for combination in CHII_COMBINATIONS[number]:
            if transformed_hand[offset + int(combination[0])-1] > 0 \
                and transformed_hand[offset + int(combination[1])-1] > 0:
                return True

        return False 

    def can_pon(self, tile) -> bool:
        transformed_hand = transform_hand(self._player_tiles)

        tile = tile[0:2]
        tile_idx = TILES.index(tile)

        return transformed_hand[tile_idx] >= 2

    def can_chakan(self) -> bool:
        transformed_hand = transform_hand(self._player_tiles)

        for tile_cnt in transformed_hand:
            if tile_cnt == 4:
                return True

        return False

    def can_kan(self, tile) -> bool:
        transformed_hand = transform_hand(self._player_tiles)

        tile = tile[0:2]
        tile_idx = TILES.index(tile)

        return transformed_hand[tile_idx] >= 3

    def can_riichi(self) -> bool:
        if self._player_scores[self._player_idx] < 10:
            return False

        if self._riichi_status[self._player_idx] == 1:
            return False

        if self._is_open_hand == True:
            return False

        transformed_hand = transform_hand(deepcopy(self._player_tiles))

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

        counts = [0, 0, 0, 0] # mentsu, pair, taatsu, isolated
        combinations = [m_combinations, p_combinations, s_combinations, w_combinations, d_combinations]
        for combination in combinations:
            for idx, count in enumerate(combination):
                counts[idx] += count 

        # check for chiitoi
        if counts[1] == 6:
            return True

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
                if tiles[idx]: tiles[idx] -= 1
                if tiles[idx + 1]: tiles[idx + 1] -= 1
                if tiles[idx + 2]: tiles[idx + 2] -= 1

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
            f"_private_tiles: {self._player_tiles}\n"
            f"_private_discarded_tiles: {self._player_discarded_tiles}\n"
            f"_others_discarded_tiles: {self._others_discarded_tiles}\n"
            f"_private_open_tiles: {self._player_open_tiles}\n"
            f"_others_open_tiles: {self._others_open_tiles}\n"
            f"_dora_indicators: {self._dora_indicators}\n"
            f"_round_name: {self._round_name}\n"
            f"_player_scores: {self._player_scores}\n"
            f"_self_wind: {self._player_wind}\n"
            f"_aka_doras_in_hand: {self._aka_doras_in_hand}\n"
            f"_riichi_status: {self._riichi_status}\n"
            "---------------------------------------\n"
        )

def transform_hand(hand):
    new_hand = [0] * 34
    mapped_hand = list(map(lambda x: x[0:2], hand))

    for tile in mapped_hand:
        idx = TILES.index(tile)
        new_hand[idx] += 1

    return new_hand

def decode_tile(tile_code):
    return TILES[tile_code // 4] + str(tile_code % 4)

def main(logs_dir):
    data = Data()
    game = Game('DEFAULT')
    
    log_paths = glob.glob(f"{logs_dir}/**/*.xml", recursive=True)
    if len(log_paths) == 0:
        print("Found no logs...")
        sys.exit(-1)

    if EXPORT_DATA:
        for key in data:
            fn = f"data/model_data/{key}_data.dat"
            if os.path.exists(fn):
                os.remove(fn)

    for i in range(0, len(log_paths), BATCH_SIZE):
        batch = log_paths[i:i+BATCH_SIZE] 

        if data.is_full():
            print("Finished collecting all data...")
            break

        for log_path in tqdm(batch):
            game.decode(open(log_path))
            game_data = game.asdata()

            parser = GameLogParser(data, game_data)
            parser.get_data()

        if EXPORT_DATA: data.export()

if __name__ == '__main__':
    logging.basicConfig(level=LOGGING_LEVEL)

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--folder", help="folder containing .xml-logs")

    args = parser.parse_args()

    main(args.folder)
