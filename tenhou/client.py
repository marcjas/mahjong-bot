from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time
import base64
import cv2
from imageio.v2 import imread
import io
import read_board
import read_scoreboard
import sys
import argparse
sys.path.append("game_models")
from models import CallModel, RiichiModel, DiscardModel

tenhou_id = "ID57255B6D-m3eZdhEG"
tenhou_url = "https://tenhou.net/3/"

def main(size_mult=1):
    options = webdriver.ChromeOptions()
    options.add_argument("--allow-file-access-from-files")
    options.add_argument("--disable-web-security")
    options.add_argument("--log-level=3")
    options.add_argument("--disable-infobars")
    options.add_experimental_option("useAutomationExtension", False)
    options.add_experimental_option("excludeSwitches", ["enable-automation"])

    driver = webdriver.Chrome(options=options)
    driver.set_window_size((read_board.expected_width * size_mult) + 16, (read_board.expected_height * size_mult) + 160)
    driver.get(tenhou_url)
    
    print("Login to continue.")
    WebDriverWait(driver, 300).until(
        EC.element_to_be_clickable((By.XPATH, "//button[@name='testplay']"))
    )

    print("Auto-adjusting window size")
    auto_adjust(driver, size_mult)
    print("Finished auto adjusting")

    while True:
        command = input("Command: ").lower()
        
        if command == "help" or command == "?":
            print("help:   This menu")
            print("?:      This menu")
            print("read:   Read the board")
            print("adjust: Auto-adjust the window size")
            print("stop:   Exit the program")
        elif command == "read":
            read(driver, size_mult)
        elif command == "adjust":
            auto_adjust(driver, size_mult)
        elif command == "save":
            save(driver)
        elif command == "html":
            f = open("page.html", "w")
            f.write(driver.page_source)
            f.close()
        elif command == "manual":
            play_manual(driver, size_mult)
        elif command == "auto":
            play_auto(driver, size_mult)
        elif command == "stop":
            driver.close()
            break

class PlayerState:

    def __init__(self):
        self.private_tiles = []
        self.discarded_tiles = [[]] * 4
        self.open_tiles = [[]] * 4
        self.dora_indicators = []
        self.round = 0
        self.player_scores = [250] * 4
        self.player_wind = 0
        self.aka_doras_in_hand = 0
        self.opponent_riichi = [False] * 3
        self.tiles = [
            "1m", "2m", "3m", "4m", 
            "5m", "6m", "7m", "8m", "9m",
            "1p", "2p", "3p", "4p", 
            "5p", "6p", "7p", "8p", "9p", 
            "1s", "2s", "3s", "4s", 
            "5s", "6s", "7s", "8s", "9s", 
            "ew", "sw", "ww", "nw",       
            "wd", "gd", "rd" 
        ]

    def update(self, board, scoreboard):
        try:
            self.private_tiles = read_board.get_player_tiles(board)
            self.discarded_tiles = read_board.get_discarded(board)
            self.open_tiles = read_board.get_open_tiles(board)
            self.dora_indicators = read_board.get_doras(board)
            self.round = self.update_round(scoreboard)
            self.player_scores = read_scoreboard.get_scores(scoreboard)
            self.player_wind = read_scoreboard.get_player_wind(scoreboard)
            self.aka_doras_in_hand = self.get_aka_doras_in_hand()
            self.opponent_riichi = [self.in_riichi(owner=i) for i in range(1,4)]
            return True
        except:
            return False

    def update_round(self, scoreboard):
        wind = read_scoreboard.get_round_wind(scoreboard)
        round_number = read_scoreboard.get_round_number(scoreboard)
        return wind * 4 + round_number

    def get_aka_doras_in_hand(self):
        return len([i for i in self.discarded_tiles[0] if i[:1] == "0"])

    def in_riichi(self, owner=0):
        return len([i for i in self.discarded_tiles[owner] if i[2:3] == "r"])

    def map_tiles(self, tiles):
        return list(map(lambda x: x[0:2].replace("0", "5"), tiles))

    def to_features_list(self):
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

        # self.private_tiles
        feature_private_tiles = [0] * 34 * 4
        mapped_private_tiles = self.map_tiles(self.private_tiles)
        for tile in mapped_private_tiles:
            idx = self.tiles.index(tile)
            for i in range(4):
                if feature_private_tiles[idx + i * 34] == 0:
                    feature_private_tiles[idx + i * 34] = 1
                    break

        # self.discarded_tiles
        feature_discarded_tiles = [0] * 34 * 30 * 4
        for player_i, discarded_tiles in enumerate(self.discarded_tiles):
            mapped_discarded_tiles = self.map_tiles(discarded_tiles)
            for i, tile in enumerate(mapped_discarded_tiles):
                idx = self.tiles.index(tile)
                feature_discarded_tiles[idx + (i * 34) + (player_i * 34 * 30)] = 1

        # self.open_tiles
        feature_open_tiles = [0] * 34 * 4 * 4
        for player_i, player_melds in enumerate(self.open_tiles):
            for i, meld in enumerate(player_melds):
                mapped_meld = self.map_tiles(meld)
                for tile in mapped_meld:
                    idx = self.tiles.index(tile)
                    for j in range(4):
                        if feature_open_tiles[idx + (j * 34 + (player_i * 34 * 4))] == 0:
                            feature_open_tiles[idx + (j * 34 + (player_i * 34 * 4))] = 1
                            break

        # self._dora_indicators
        feature_dora_indicators = [0] * 34 * 4
        mapped_dora_indicators = self.map_tiles(self.dora_indicators)
        for tile in mapped_dora_indicators:
            idx = self.tiles.index(tile)
            for i in range(4):
                if feature_dora_indicators[idx + i * 34] == 0:
                    feature_dora_indicators[idx + i * 34] = 1
                    break

        # self._round_name
        round_name = [0] * 12
        round_name[self.round] = 1

        # self._player_scores
        player_scores = [0] * 4
        for i, player_score in enumerate(self.player_scores):
            if player_score >= 1000:
                player_scores[i] = 1
            elif player_score <= 0:
                player_scores[i] = 0
            else:
                player_scores[i] = player_score / 1000

        # self._self_wind
        self_wind = [0] * 4
        self_wind[self.player_wind] = 1

        # self._aka_doras_in_hand
        aka_doras_in_hand = [0] * 4
        aka_doras_in_hand[self.aka_doras_in_hand] = 1

        # self._riichi_status
        others_riichi_status = self.opponent_riichi

        return feature_private_tiles + \
               feature_discarded_tiles + \
               feature_open_tiles + \
               feature_dora_indicators + \
               round_name + \
               player_scores + \
               self_wind + \
               aka_doras_in_hand + \
               others_riichi_status

def play_manual(driver, size_mult=1):
    new_round = False
    while True:
        time.sleep(0.5)
        canvases = driver.find_elements(By.TAG_NAME, "canvas")
        scoreboard = decode_canvas(driver, canvases[2])
        scoreboard = cv2.cvtColor(scoreboard, cv2.COLOR_RGB2BGR)
        
        reset_cursor(driver)

        turn_player = read_scoreboard.get_turn_player(scoreboard)
        if turn_player == -1:
            print("Not in an active round. Waiting.")
            new_round = True
            time.sleep(1)
            continue
        if new_round:
            new_round = False
            time.sleep(5)
            canvases = driver.find_elements(By.TAG_NAME, "canvas")
            scoreboard = decode_canvas(driver, canvases[2])
            scoreboard = cv2.cvtColor(scoreboard, cv2.COLOR_RGB2BGR)
            turn_player = read_scoreboard.get_turn_player(scoreboard)

        board = decode_canvas(driver, canvases[1])
        board = cv2.cvtColor(board, cv2.COLOR_RGB2BGR)
        
        discarded = read_board.get_discarded(board)
        riichi = len([i for i in discarded[0] if i[2:3] == "r"])

        if not riichi and turn_player == 0:
            time.sleep(0.5)
            if find_visible_button(driver, "Kan") is not None:
                choice = input("Closed Kan? (y/n): ")
                do_call(driver, "Kan" if choice == "y" else "×")
            if find_visible_button(driver, "Riichi") is not None:
                choice = input("Riichi? (y/n): ")
                if choice == "y":
                    do_call(driver, "Riichi")
                    riichi = True
            if find_visible_button(driver, "Tsumo") is not None:
                choice = "y" #input("Tsumo? (y/n): ")
                if choice == "y":
                    do_call(driver, "Tsumo")
            hand = read_board.get_player_tiles(board)
            discard_tile = input("Tile to discard: ")
            while discard_tile not in hand and discard_tile != "skip":
                print("You do not have that tile.")
                discard_tile = input("Tile to discard: ")
            if discard_tile != "skip":
                do_discard(driver, hand.index(discard_tile), size_mult=size_mult)
            while read_scoreboard.get_turn_player(scoreboard) == 0:
                canvases = driver.find_elements(By.TAG_NAME, "canvas")
                scoreboard = decode_canvas(driver, canvases[2])
                scoreboard = cv2.cvtColor(scoreboard, cv2.COLOR_RGB2BGR)
                time.sleep(0.5)
        elif find_visible_button(driver, "Chii") is not None:
            time.sleep(0.2)
            tile = [i for i in discarded[turn_player] if len(i) > 2 and "c" in i[2:]][0][:2]
            choice = chii_model.predict(player_state.to_features_list())
            do_call(driver, "Chii" if choice == "y" else "×")
        elif find_visible_button(driver, "Pon") is not None:
            time.sleep(0.2)
            tile = [i for i in discarded[turn_player] if len(i) > 2 and "c" in i[2:]][0][:2]
            choice = input(f"Pon {tile}? (y/n): ")
            do_call(driver, "Pon" if choice == "y" else "×")
        elif find_visible_button(driver, "Kan") is not None:
            time.sleep(0.2)
            tile = [i for i in discarded[turn_player] if len(i) > 2 and "c" in i[2:]][0][:2]
            choice = input(f"Kan {tile}? (y/n): ")
            do_call(driver, "Kan" if choice == "y" else "×")
        elif find_visible_button(driver, "Call") is not None:
            time.sleep(0.2)
            tile = [i for i in discarded[turn_player] if len(i) > 2 and "c" in i[2:]][0][:2]
            choice = input(f"Call {tile}? (y/n): ")
            do_call(driver, "Call" if choice == "y" else "×", manual=True)
        elif find_visible_button(driver, "Ron") is not None:
            time.sleep(0.2)
            choice = "y" #input("Ron? (y/n): ")
            do_call(driver, "Ron" if choice == "y" else "×", manual=True)
        if find_visible_button(driver, "Tsumo") is not None:
            time.sleep(0.2)
            choice = "y" #input("Tsumo? (y/n): ")
            if choice == "y":
                do_call(driver, "Tsumo")

def play_auto(driver, size_mult=1):
    player_state = PlayerState()

    chii_model = CallModel("models/chii_model.pt")
    pon_model = CallModel("models/pon_model.pt")
    kan_model = CallModel("models/kan_model.pt")
    riichi_model = RiichiModel("models/riichi_model.pt")
    discard_model = DiscardModel("models/discard_model.pt")

    new_round = False
    while True:
        time.sleep(0.5)
        canvases = driver.find_elements(By.TAG_NAME, "canvas")
        scoreboard = decode_canvas(driver, canvases[2])
        scoreboard = cv2.cvtColor(scoreboard, cv2.COLOR_RGB2BGR)
        
        reset_cursor(driver)

        turn_player = read_scoreboard.get_turn_player(scoreboard)
        if turn_player == -1:
            if not new_round:
                print("Not in an active round. Waiting.")
            new_round = True
            time.sleep(1)
            continue
        if new_round:
            new_round = False
            time.sleep(5)
            canvases = driver.find_elements(By.TAG_NAME, "canvas")
            scoreboard = decode_canvas(driver, canvases[2])
            scoreboard = cv2.cvtColor(scoreboard, cv2.COLOR_RGB2BGR)
            turn_player = read_scoreboard.get_turn_player(scoreboard)

        board = decode_canvas(driver, canvases[1])
        board = cv2.cvtColor(board, cv2.COLOR_RGB2BGR)

        if turn_player == 0:
            if player_state.update(board, scoreboard):
                if not player_state.in_riichi():
                    time.sleep(0.5)
                    if find_visible_button(driver, "Tsumo") is not None:
                        should_tsumo = True #input("Tsumo? (y/n): ")
                        if should_tsumo:
                            do_call(driver, "Tsumo")
                    if find_visible_button(driver, "Kan") is not None:
                        should_call = kan_model.predict(player_state.to_features_list())
                        do_call(driver, "Kan" if should_call else "×")
                    if find_visible_button(driver, "Riichi") is not None:
                        should_riichi = riichi_model.predict(player_state.to_features_list())
                        if should_riichi:
                            do_call(driver, "Riichi")
                    hand = read_board.get_player_tiles(board)
                    discard_tile = discard_model.predict(player_state.to_features_list())
                    if discard_tile not in hand:
                        discard_tile = discard_tile.replace("5","0")
                    if discard_tile not in hand:
                        print("You do not have that tile.")
                    else:
                        print("Discarding", discard_tile)
                        do_discard(driver, hand.index(discard_tile), size_mult=size_mult)
                    while read_scoreboard.get_turn_player(scoreboard) == 0:
                        canvases = driver.find_elements(By.TAG_NAME, "canvas")
                        scoreboard = decode_canvas(driver, canvases[2])
                        scoreboard = cv2.cvtColor(scoreboard, cv2.COLOR_RGB2BGR)
                        time.sleep(0.5)
        elif find_visible_button(driver, "Ron") is not None:
            time.sleep(0.2)
            if player_state.update(board, scoreboard):
                should_ron = True #input("Ron? (y/n): ")
                do_call(driver, "Ron" if should_ron else "×", manual=True)
        elif find_visible_button(driver, "Chii") is not None:
            time.sleep(0.2)
            if player_state.update(board, scoreboard):
                #tile = [i for i in discarded[turn_player] if len(i) > 2 and "c" in i[2:]][0][:2]
                should_call = chii_model.predict(player_state.to_features_list())
                do_call(driver, "Chii" if should_call else "×")
        elif find_visible_button(driver, "Pon") is not None:
            time.sleep(0.2)
            if player_state.update(board, scoreboard):
                #tile = [i for i in discarded[turn_player] if len(i) > 2 and "c" in i[2:]][0][:2]
                should_call = pon_model.predict(player_state.to_features_list())
                do_call(driver, "Pon" if should_call else "×")
        elif find_visible_button(driver, "Kan") is not None:
            time.sleep(0.2)
            if player_state.update(board, scoreboard):
                #tile = [i for i in player_state.discarded_tiles[turn_player] if len(i) > 2 and "c" in i[2:]][0][:2]
                should_call = kan_model.predict(player_state.to_features_list())
                do_call(driver, "Kan" if should_call else "×")
        elif find_visible_button(driver, "Call") is not None:
            time.sleep(0.2)
            if player_state.update(board, scoreboard):
                #tile = [i for i in discarded[turn_player] if len(i) > 2 and "c" in i[2:]][0][:2]
                should_call = chii_model.predict(player_state.to_features_list())
                do_call(driver, "Call" if should_call else "×", manual=True)
        if find_visible_button(driver, "Tsumo") is not None:
            time.sleep(0.2)
            if player_state.update(board, scoreboard):
                should_tsumo = True #input("Tsumo? (y/n): ")
                if should_tsumo:
                    do_call(driver, "Tsumo")

def reset_cursor(driver):
    canvases = driver.find_elements(By.TAG_NAME, "canvas")
    ac = ActionChains(driver)
    ac.move_to_element_with_offset(canvases[1], 6, 6).perform()

def do_discard(driver, index, size_mult=1):
    canvases = driver.find_elements(By.TAG_NAME, "canvas")
    ac = ActionChains(driver)
    ac.move_to_element_with_offset(canvases[1],
        36 + 40 * index,
        read_board.expected_height - 46) \
        .click().perform()

def do_call(driver, action, manual=False):
    try:
        element = find_visible_button(driver, action)
        ac = ActionChains(driver)
        ac.move_to_element(element).click().perform()
        if action == "Call":
            time.sleep(0.2)
            elements = driver.find_elements(By.CLASS_NAME, "bgb")
            visible_elements = []
            for element in elements:
                parent = element.find_element(By.XPATH, "..")
                style = parent.get_attribute("style")
                if "visibility: hidden" not in style and "display: none" not in style:
                    visible_elements.append(element)
            if len(visible_elements) > 0:
                ac = ActionChains(driver)
                ac.move_to_element(visible_elements[0]).click().perform()
            else:
                print("Found no Call options, this should not happen.")
    except:
        print("Failed to click on element")

def find_visible_button(driver, action):
    bold = (action == "Ron" or action == "Tsumo")
    bold_str = "/b" if bold else ""
    xpath = f"//div{bold_str}[text()[contains(., '{action}')]]"
    if not try_find_element(driver, xpath):
        return None
    elements = driver.find_elements(By.XPATH, xpath)
    for element in elements:
        parent_path = "../.." if bold else ".."
        parent = element.find_element(By.XPATH, parent_path)
        style = parent.get_attribute("style")
        if "visibility: hidden" not in style and "display: none" not in style:
            return element
    return None

def try_find_element(driver, xpath):
    try:
        driver.find_element(By.XPATH, xpath)
        return True
    except:
        return False

def read(driver, size_mult=1):
    canvases = driver.find_elements(By.TAG_NAME, "canvas")
    board = decode_canvas(driver, canvases[1])
    board = cv2.cvtColor(board, cv2.COLOR_RGB2BGR)
    scoreboard = decode_canvas(driver, canvases[2])
    scoreboard = cv2.cvtColor(scoreboard, cv2.COLOR_RGB2BGR)

    if board.shape[1] != read_board.expected_width * size_mult:
        print(f"Window size was changed. Auto-adjusting...")
        auto_adjust(driver, size_mult)
        canvases = driver.find_elements(By.TAG_NAME, "canvas")
        board = decode_canvas(driver, canvases[1])
        board = cv2.cvtColor(board, cv2.COLOR_RGB2BGR)

    if board.shape[1] != read_board.expected_width:
        board = cv2.resize(board, (read_board.expected_width, read_board.expected_height))

    if scoreboard.shape[1] != read_scoreboard.expected_width:
        scoreboard = cv2.resize(scoreboard, (read_scoreboard.expected_width, read_scoreboard.expected_height))
    
    player_tiles = read_board.get_player_tiles(board)
    print(' '.join(player_tiles))

    discarded_tiles = read_board.get_discarded(board)
    print("Discarded:")
    for i in range(4):
        print(i, ":", ' '.join(discarded_tiles[i]))

    doras = read_board.get_doras(board)
    print("Doras:", ' '.join(doras))

    open_tiles = read_board.get_open_tiles(board)
    for i in range(4):
        print(i, ":", open_tiles[i])

    scores = read_scoreboard.get_scores(scoreboard)
    print("Scores:", scores)
    
def save(driver):
    canvases = driver.find_elements(By.TAG_NAME, "canvas")

    board = decode_canvas(driver, canvases[1])
    board = cv2.cvtColor(board, cv2.COLOR_RGB2BGR)
    cv2.imwrite("board.png", board)

    scoreboard = decode_canvas(driver, canvases[2])
    scoreboard = cv2.cvtColor(scoreboard, cv2.COLOR_RGB2BGR)
    cv2.imwrite("scoreboard.png", scoreboard)

def decode_canvas(driver, canvas):
    canvas_base64 = driver.execute_script("return arguments[0].toDataURL('image/png').substring(21);", canvas)
    canvas_png = base64.b64decode(canvas_base64)
    return imread(io.BytesIO(canvas_png))

def auto_adjust(driver, size_mult=1):
    while True:
        time.sleep(0.5)

        canvases = driver.find_elements(By.TAG_NAME, "canvas")
        board = decode_canvas(driver, canvases[1])
        window_size = driver.get_window_size()

        if board.shape[1] < read_board.expected_width * size_mult or \
           board.shape[0] < read_board.expected_height * size_mult:
            window_size["width"] += 1
            window_size["height"] += 1
        elif board.shape[1] > read_board.expected_width * size_mult or \
             board.shape[0] < read_board.expected_height * size_mult:
            window_size["width"] -= 1
        else:
            break

        driver.set_window_size(window_size["width"], window_size["height"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start an automated Chrome browser with Tenhou bot functionality.")
    parser.add_argument("-S", "--sizemult", dest="size_mult", type=int, default=1, help="Apply a modifier to the screen size (default: 1)")
    args = parser.parse_args()

    if args.size_mult < 1:
        print("sizemult must be at least 1")
        sys.exit(-1)

    main(size_mult=args.size_mult)