import logging
import time
import random
import wandb
from models import CallModel, RiichiModel, DiscardModel

class Controller:
    def __init__(self, client, logging_level=logging.INFO):
        logging.basicConfig(level=logging_level)

        self.client        = client
        self.chii_model    = CallModel("models/chii_model.pt")
        self.pon_model     = CallModel("models/pon_model.pt")
        self.kan_model     = CallModel("models/kan_model.pt")
        self.riichi_model  = RiichiModel("models/riichi_model.pt")
        self.discard_model = DiscardModel("models/discard_model.pt")

    def play(self):
        new_round = False

        game_done_timer = 0.0

        while True:
            time.sleep(0.1)
            
            if self.can_redeal():
                self.client.do_action("Redeal")
                logging.info(f"Redealt the hand.")
                time.sleep(5.0)

            turn_player = self.client.get_turn_player()

            if turn_player is None:
                if not new_round:
                    logging.info("Not in active round. Waiting...")
                new_round = True
            else:
                if new_round:
                    logging.info(f"Starting {self.client.get_round_name()}")
                    new_round = False
                    time.sleep(5.0)

                if turn_player == 0:
                    time.sleep(0.5 + random.random())
                    self.do_turn()
                    time.sleep(0.1)
                    self.client.reset_cursor()
                    while self.client.get_turn_player() == 0:
                        time.sleep(0.1)
                        game_done_timer += 0.1
                        # Game must have ended if 30 seconds has passed
                        if game_done_timer > 30:
                            break
                elif self.can_call() or self.client.find_visible_button("Call"):
                    time.sleep(0.5 + random.random())
                    self.do_call()
                    time.sleep(0.1)
                    self.client.reset_cursor()

            if self.client.find_ok_button() is not None:
                game_done_timer += 0.1
                # Game has ended if results show for more than 15 seconds
                if game_done_timer > 15:
                    logging.info("Game finished.")
                    self.client.find_ok_button().click()
                    return
            else:
                game_done_timer = 0.0


    def do_turn(self):
        player_state = self.client.get_player_state()

        if self.can_win():
            if self.should_win(player_state):
                self.win()

        if self.can_kan():
            if self.should_kan(player_state):
                self.client.do_action("Kan")
                logging.info(f"Closed Kan")
                time.sleep(2.0)
                self.do_turn()
                return

        if player_state.in_riichi():
            return

        if self.can_riichi():
            if self.should_riichi(player_state):
                self.client.do_action("Riichi")
                logging.info(f"Declared Riichi!")
                time.sleep(2.0)

        discard_tile = self.get_discard_tile(player_state)

        if discard_tile not in player_state.private_tiles:
            discard_tile = discard_tile.replace("5", "0")
        if discard_tile not in player_state.private_tiles:
            logging.error(f"There is no private {discard_tile}")
            logging.error(f"Private tiles: {' '.join(player_state.private_tiles)}")
        else:
            discard_tile_index = player_state.private_tiles.index(discard_tile)
            self.client.do_discard(discard_tile_index)
            logging.info(f"Discarded {discard_tile}")

    def do_call(self):
        player_state = self.client.get_player_state()
        called_tile = player_state.get_called_tile()

        #if called_tile is None:
        #    print(player_state)
        #    self.client.save_board_image(file_name=f"error_board_{int(time.time())}")

        if self.can_win():
            if self.should_win(player_state):
                self.win()
                return

        if self.client.find_visible_button("Call"):
            self.client.do_action("Call")
            time.sleep(1.0)
            meld_options = self.client.get_meld_options()
            if len(meld_options) == 0:
                logging.error("Got an empty meld option list.")
            else:
                for i in meld_options:
                    i.append(called_tile)
                choice = self.choose_meld_option(player_state, meld_options)
                if choice >= 0:
                    self.client.do_meld_option(choice)
                    logging.info(f"Called {called_tile}")
                    time.sleep(1.0)
                    return
                else:
                    self.client.do_action("×")
                    time.sleep(1.0)
                    self.client.do_action("×")
        else:
            if self.can_chii():
                if self.should_chii(player_state):
                    self.client.do_action("Chii")
                    logging.info(f"Called Chii on {called_tile}")
                    time.sleep(1.0)
                    return
                else:
                    self.client.do_action("×")

            if self.can_pon():
                if self.should_pon(player_state):
                    self.client.do_action("Pon")
                    logging.info(f"Called Pon on {called_tile}")
                    time.sleep(1.0)
                    return
                else:
                    self.client.do_action("×")

            if self.can_kan():
                if self.should_kan(player_state):
                    self.client.do_action("Kan")
                    logging.info(f"Called Kan on {called_tile}")
                    time.sleep(1.0)
                    return
                else:
                    self.client.do_action("×")

        logging.info(f"Did not Call for {called_tile}")

    def win(self):
        if self.can_ron():
            self.client.do_action("Ron")
            logging.info(f"Declared Win by Ron!")
            time.sleep(1.0)
        elif self.can_tsumo():
            self.client.do_action("Tsumo")
            logging.info(f"Declared Win by Tsumo!")
            time.sleep(1.0)

    def can_win(self):
        return self.can_ron() or self.can_tsumo()

    def can_ron(self):
        return self.client.find_visible_button("Ron")

    def can_tsumo(self):
        return self.client.find_visible_button("Tsumo")

    def can_call(self):
        return self.can_chii() or self.can_pon() or self.can_kan() or self.can_ron()

    def can_chii(self):
        return self.client.find_visible_button("Chii")

    def can_pon(self):
        return self.client.find_visible_button("Pon")

    def can_kan(self):
        return self.client.find_visible_button("Kan")

    def can_riichi(self):
        return self.client.find_visible_button("Riichi")

    def can_redeal(self):
        return self.client.find_visible_button("Redeal")

    def should_win(self, player_state):
        pass

    def should_kan(self, player_state):
        pass

    def should_riichi(self, player_state):
        pass

    def should_chii(self, player_state):
        pass

    def should_pon(self, player_state):
        pass

    def choose_meld_option(self, player_state, meld_options):
        pass

    def get_discard_tile(self, player_state):
        pass

class ControllerAuto(Controller):

    USE_WANDB = False

    def __init__(self, client, total_games=1, logging_level=logging.INFO):
        super().__init__(client, logging_level=logging_level)
        self.game_number = 0
        self.total_games = total_games
        if self.USE_WANDB: 
            config = {
                "VERSION": 1
            }
            wandb.init(project="riichi-mahjong", entity="shuthus", tags=["PlayData"], config=config)

    def play(self):
        if self.total_games > 0:
            for i in range(self.total_games):
                logging.info(f"Game {i + 1} / {self.total_games}")
                self.play_game(self.game_number)
        else:
            while True:
                logging.info(f"Game {self.game_number + 1}")
                self.play_game(self.game_number)

    def play_game(self, game_number):
        #self.client.join_game()
        time.sleep(1.0)
        super().play()
        time.sleep(5.0)
        self.process_result(self.client.get_final_player_scores())
        time.sleep(5.0)
        self.game_number += 1
        if self.total_games <= 0 or self.game_number < self.total_games:
            self.client.find_ok_button().click()
            time.sleep(15.0)

    def should_kan(self, player_state):
        score = self.kan_model.predict_raw(player_state.to_feature_list())
        logging.info(f"Score on kan: {score}")
        return (score >= 0.5)

    def should_riichi(self, player_state):
        score = self.riichi_model.predict_raw(player_state.to_feature_list())
        logging.info(f"Score on riichi: {score}")
        return (score >= 0.5)

    def should_chii(self, player_state):
        score = self.chii_model.predict_raw(player_state.to_feature_list())
        logging.info(f"Score on chii: {score}")
        return (score >= 0.5)

    def should_pon(self, player_state):
        score = self.pon_model.predict_raw(player_state.to_feature_list())
        logging.info(f"Score on pon: {score}")
        return (score >= 0.5)

    def should_win(self, player_state):
        return True

    def choose_meld_option(self, player_state, meld_options):
        scores = []
        for meld in meld_options:
            meld_type = self.get_meld_type(meld)
            if meld_type == "kan":
                score = self.kan_model.predict_raw(player_state.to_feature_list())
                scores.append(score)
            elif meld_type == "chii":
                score = self.chii_model.predict_raw(player_state.to_feature_list())
                scores.append(score)
            elif meld_type == "pon":
                score = self.pon_model.predict_raw(player_state.to_feature_list())
                scores.append(score)
            else:
                logging.error("Got an invalid meld type")
                score = 0.0
                scores.append(score)
            logging.info(f"Score on {meld_type} {str(meld)}: {score}")

        best_index = scores.index(max(scores))
        if scores[best_index] >= 0.5:
            return best_index
        else:
            return -1

    def get_meld_type(self, meld):
        if len(meld) == 4:
            return "kan"
        meld = [tile.replace("0", "5") for tile in meld]
        for i in range(1, len(meld)):
            if meld[0] != meld[i]:
                return "chii"
        return "pon"

    def get_discard_tile(self, player_state):
        return self.discard_model.predict(player_state.to_feature_list())

    def process_result(self, final_scores):
        player_score = [i for i in final_scores if i[0] == self.client.user_name][0][1]
        placing = 0
        for name, score in final_scores:
            placing += 1
            if player_score == score:
                break
        if self.USE_WANDB:
            wandb.log({
                "Game": self.game_number,
                "Placing": placing,
                "Final Score": player_score
            })

class ControllerManual(Controller):
    def __init__(self, client, logging_level=logging.INFO):
        super().__init__(client, logging_level=logging_level)

    def should_kan(self, player_state):
        return (input("Should Kan? (y/n): ") == "y")

    def should_riichi(self, player_state):
        return (input("Should Riichi? (y/n): ") == "y")

    def should_chii(self, player_state):
        return (input("Should Chii? (y/n): ") == "y")

    def should_pon(self, player_state):
        return (input("Should Pon? (y/n): ") == "y")

    def should_win(self, player_state):
        return (input("Should Win? (y/n): ") == "y")

    def choose_meld_option(self, player_state, meld_options):
        print("Choose a meld option...")
        for i in meld_options:
            print(f" {i}): {' '.join(i)}")
        print(" Other: Don't Meld")
        try:
            choice = int(input("Meld choice: "))
            if choice >= 0 and choice < len(meld_options):
                return choice
            else:
                return -1
        except:
            return -1

    def get_discard_tile(self, player_state):
        return input("Tile to discard: ")