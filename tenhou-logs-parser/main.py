from typing import Dict
from attr import asdict
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
        pass

    def kan_model_data(self):
        pass

    def riichi_model_data(self):
        pass

    def discard_model_data(self):
        states = []
        actions = []

        player_scores = [25000, 25000, 25000, 25000]

        for index, round in enumerate(self._rounds):
            # Initial state  
            round_state = RoundState(round, player_scores)

            for event in round["events"]:
                if event["type"] == "Discard":
                    player_idx = event["player"]
                    states.append(round_state._player_states[player_idx].to_model_input())
                    actions.append(event["tile"])

                round_state.update_state(event)

            if round["agari"]:
                round_state.update_player_scores(round["agari"])
                player_scores = round_state._player_scores
            # todo sjekk honba logikk
            
            break

        # returner (states, actions) med pickle (?)


tiles = [
    "0m", "1m", "2m", "3m", "4m", 
    "5m", "6m", "7m", "8m", "9m",
    "0p", "1p", "2p", "3p", "4p", 
    "5p", "6p", "7p", "8p", "9p", 
    "0s", "1s", "2s", "3s", "4s", 
    "5s", "6s", "7s", "8s", "9s", 
    "ew", "sw", "ww", "nw",       
    "wd", "gd", "rd"              
]

m_tiles = tiles[1:10]
p_tiles = tiles[11:20]
s_tiles = tiles[21:30]

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
        winds = ["east", "north", "west", "south"]
        return winds[(player_idx - dealer_idx) % 4]

    def update_player_scores(self, agari):
        pass

    def update_state(self, event):
        match event["type"]:
            case "Dora":
                self._dora_indicators.append(event["tile"][0:2])

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
                pass

        self.update_players_aka_dora_count_in_hand()

    def update_players_aka_dora_count_in_hand(self):
        self._p0_state.update_aka_dora_count_in_hand()
        self._p1_state.update_aka_dora_count_in_hand()
        self._p2_state.update_aka_dora_count_in_hand()
        self._p3_state.update_aka_dora_count_in_hand()

        

    
class PlayerObservableState:
    def __init__(self, player_idx, round, player_scores, discarded_tiles, open_tiles, dora_indicators, wind, riichi_status):
        self._player_idx = player_idx 
        self._discarded_tiles = discarded_tiles
        self._open_tiles = open_tiles
        self._round_info = round["round"] # Tuple of string (name), int (honba count), int (leftover riichi sticks)
        self._turn = 0
        self._riichi_status = riichi_status

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
        self._others_open_tiles = self._discarded_tiles[:self._player_idx] + self._discarded_tiles[self._player_idx + 1:]
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
        self._discarded_tiles.append(tile)

        self._turn += 1

    def draw(self, tile):
        self._private_tiles.append(tile)

    def to_model_input(self):
        return self._private_tiles


def main(logs_dir = "../tenhou-logs/xml-logs"):
    game = Game('DEFAULT')

    log_paths = [logs_dir + f for f in os.listdir(logs_dir)]

    for log_path in log_paths:
        print(log_path)
        game.decode(open(log_path))

        game_data = game.asdata()

        model_input = Dataset(game_data) 

        model_input.discard_model_data()

        # break


if __name__ == '__main__':
    args = sys.argv[1:]

    if len(args) == 0:
        main()
    elif len(args) == 1:
        main(args[0])
    else:
        print("invalid command")