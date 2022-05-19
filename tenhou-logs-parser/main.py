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
            state = State(round, player_scores)


            for event in round["events"]:
                
                state.update_state(event)


                break
            
            if round["agari"]:
                state.update_player_scores(round["agari"])
                player_scores = state._player_scores
            # todo sjekk honba logikk


            break

        # returner (states, actions) med pickle (?)

class State:
    def __init__(self, round, player_scores):
        self._round = round
        self._player_scores = player_scores
        self._discarded_tiles = []
        self._doras = []

        self._p0_state = PlayerState(0, self._round, self._player_scores, self._discarded_tiles, self._doras)
        self._p1_state = PlayerState(1, self._round, self._player_scores, self._discarded_tiles, self._doras)
        self._p2_state = PlayerState(2, self._round, self._player_scores, self._discarded_tiles, self._doras)
        self._p3_state = PlayerState(3, self._round, self._player_scores, self._discarded_tiles, self._doras)

    def update_player_scores(self, agari):
        pass

    def update_state(self, event):
        match event["type"]:
            case "Dora":
                self._doras.append(event["tile"])

            case "Discard":
                pass

            case "Draw":
                pass

            case "Call":
                pass

    
class PlayerState:
    def __init__(self, playerIdx, round, player_scores, discarded_tiles, doras):
        self._playerIdx = playerIdx 
        self._private_tiles = round["hands"][self._playerIdx]
        self._doras = doras
        self._round = round["round"]
        self._player_scores = player_scores
        self._discarded_tiles = discarded_tiles

        print(self._private_tiles)
        print(self._doras)

    def get_score(self):
        return self._player_scores[self._playerIdx]


def main(logs_dir = "../tenhou-logs/xml-logs"):
    game = Game('DEFAULT')

    log_paths = [logs_dir + f for f in os.listdir(logs_dir)]

    for log_path in log_paths:
        game.decode(open(log_path))

        game_data = game.asdata()
        
        model_input = Dataset(game_data) 

        model_input.discard_model_input()

        break


if __name__ == '__main__':
    args = sys.argv[1:]

    if len(args) == 0:
        main()
    elif len(args) == 1:
        main(args[0])
    else:
        print("invalid command")