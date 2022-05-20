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
            # initial state  
            round_state = RoundState(round, player_scores)

            for event in round["events"]:
                # if event["type"] == "Discard":
                #     states.append(state)
                #     actions.append(event["tile"])

                round_state.update_state(event)

                print(round_state._dora_indicators)

                break
            
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
wind_tiles = tiles[30:34]
dragon_tiles = tiles[34:37]



class RoundState:
    def __init__(self, round, player_scores):
        self._round = round 
        self._dealer = self._round["dealer"]
        self._player_scores = player_scores
        self._discarded_tiles = [[], [], [], []]
        self._open_tiles = [[], [], [], []]
        self._dora_indicators = []
        self._actual_doras = ["5m0", "5s0", "5p0"]

        self._p0_state = PlayerObservableState(0, self._round, self._player_scores, self._discarded_tiles, self._open_tiles, self._dora_indicators, self.get_player_wind(0, self._dealer))
        self._p1_state = PlayerObservableState(1, self._round, self._player_scores, self._discarded_tiles, self._open_tiles, self._dora_indicators, self.get_player_wind(1, self._dealer))
        self._p2_state = PlayerObservableState(2, self._round, self._player_scores, self._discarded_tiles, self._open_tiles, self._dora_indicators, self.get_player_wind(2, self._dealer))
        self._p3_state = PlayerObservableState(3, self._round, self._player_scores, self._discarded_tiles, self._open_tiles, self._dora_indicators, self.get_player_wind(3, self._dealer))

    def get_player_wind(self, player_idx, dealer_idx):
        winds = ["east", "north", "west", "south"]
        return winds[(player_idx - dealer_idx) % 4]

    def update_player_scores(self, agari):
        pass

    def update_state(self, event):
        match event["type"]:
            case "Dora":
                tile = event["tile"][0:2]

                self._dora_indicators.append(tile)



                tile_type = tile[1:2]

                if (tile_type == "m"):
                    print("TEST")


                # self._actual_doras.append(event["tile"])

            case "Discard":
                pass

            case "Draw":
                pass

            case "Call":
                pass

    
class PlayerObservableState:
    def __init__(self, player_idx, round, player_scores, discarded_tiles, open_tiles, dora_indicators, wind):
        self._player_idx = player_idx 
        self._discarded_tiles = discarded_tiles
        self._open_tiles = open_tiles
        self._round_info = round["round"] # Tuple of string (name), int (honba count), int (leftover riichi sticks)

        """
        Features:
            - self._private_tiles               34 x 4 x 1
            - self._private_discarded tiles     34 x 30
            - self._others_discarded_tiles      34 x 30 x 3
            - self._private_open_tiles          --
            - self._dora_indicators             34 x 5
            - self._round_name                  one hot encoding 16length (4e, 4s, ...)
            - self._player_scores               4 channels scale from 0-1 where 0=0 and 1=100.000  [0.25, 0.25, 0.25, 0.25]
        """
        self._private_tiles = round["hands"][self._player_idx]
        self._private_discarded_tiles = self._discarded_tiles[self._player_idx]
        self._others_discarded_tiles = self._discarded_tiles[:self._player_idx] + self._discarded_tiles[self._player_idx + 1:]
        self._private_open_tiles = self._open_tiles[self._player_idx] 
        self._others_open_tiles = self._discarded_tiles[:self._player_idx] + self._discarded_tiles[self._player_idx + 1:]
        self._dora_indicators = dora_indicators
        self._round_name = self._round_info[0]
        # TODO self._round_leftover_riichi_sticks = self._round_info[2]
        self._player_scores = player_scores
        self._self_wind = wind

        # print(self._private_tiles)
        # print(self._doras)

    def get_score(self):
        return self._player_scores[self._player_idx]


def main(logs_dir = "../tenhou-logs/xml-logs"):
    game = Game('DEFAULT')

    log_paths = [logs_dir + f for f in os.listdir(logs_dir)]

    for log_path in log_paths:
        game.decode(open(log_path))

        game_data = game.asdata()

        model_input = Dataset(game_data) 

        model_input.discard_model_data()

        break


if __name__ == '__main__':
    args = sys.argv[1:]

    if len(args) == 0:
        main()
    elif len(args) == 1:
        main(args[0])
    else:
        print("invalid command")