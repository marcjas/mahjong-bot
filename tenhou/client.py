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
import logging
import traceback
sys.path.append("game_models")
from controller import ControllerAuto, ControllerManual

tenhou_id = "ID57255B6D-m3eZdhEG"
tenhou_url = "https://tenhou.net/3/"

def main(size_mult=1):
    client = Client(logging_level=logging.INFO)
    client.start()
    client.login()

    print("Auto-adjusting window size")
    client.auto_adjust_size()
    print("Finished auto adjusting")

    while True:
        command = input("Command: ").lower().split(" ")
        
        if command[0] == "help" or command[0] == "?":
            print("help:   This menu")
            print("?:      This menu")
            print("auto N: Make AI play N games (default=1, 0=infinite)")
            print("manual: Play using the command line")
            print("read:   Read the board")
            print("save:   Save board and scoreboard to file")
            print("html:   Save raw HTML to file")
            print("adjust: Auto-adjust the window size")
            print("stop:   Exit the program")
        elif command[0] == "read":
            print(client.get_player_state())
            meld_options = client.get_meld_options()
            if len(meld_options) > 0:
                print("Meld Options:")
                for meld in meld_options:
                    print("   ", meld)
        elif command[0] == "adjust":
            client.auto_adjust_size()
        elif command[0] == "save":
            client.save_board_image()
            client.save_scoreboard_image()
            client.save_meld_images()
        elif command[0] == "html":
            with open("page.html", "w") as f:
                f.write(client.get_page_source())
        elif command[0] == "manual":
            controller = ControllerManual(client)
            controller.play()
        elif command[0] == "auto":
            print("Loading models")
            num_games = int(command[1]) if (len(command) > 1) else 1
            controller = ControllerAuto(client, total_games=num_games)
            print("Ready to Play")
            controller.play()
        elif command[0] == "stop":
            client.stop()
            break

class Client:

    def __init__(self, logging_level=logging.INFO, size_mult=1):
        self.board_reader = read_board.BoardReader()
        self.scoreboard_reader = read_scoreboard.ScoreboardReader()
        self.driver = None
        self.user_name = "NoName"
        self.size_mult = size_mult
        logging.basicConfig(level=logging_level)

    def start(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--allow-file-access-from-files")
        options.add_argument("--disable-web-security")
        options.add_argument("--log-level=3")
        options.add_argument("--disable-infobars")
        options.add_experimental_option("useAutomationExtension", False)
        options.add_experimental_option("excludeSwitches", ["enable-automation"])

        self.driver = webdriver.Chrome(options=options)
        self.driver.set_window_size(
            (read_board.expected_width * self.size_mult) + 16,
            (read_board.expected_height * self.size_mult) + 160
        )
        self.driver.get(tenhou_url)

    def login(self, user_id=None):
        element = WebDriverWait(self.driver, 300).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@name='ok']"))
        ).click()
        time.sleep(1.0)
        if user_id is None:
            WebDriverWait(self.driver, 300).until(
                EC.element_to_be_clickable((By.XPATH, "//button[text()[contains(., 'Guest Login')]]"))
            ).click()

    def join_game(self):
        while True:
            try:
                element = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[@id='join1']"))
                ).click()
                break
            except:
                element = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[@name='paneNext']"))
                ).click()
                time.sleep(1.0)

    def get_board_canvas(self):
        return self.driver.find_elements(By.TAG_NAME, "canvas")[1]

    def get_scoreboard_canvas(self):
        return self.driver.find_elements(By.TAG_NAME, "canvas")[2]

    def get_meld_canvases(self):
        all_buttons = self.driver.find_elements(By.CLASS_NAME, "bgb")

        visible_buttons = []
        for element in all_buttons:
            parent = element.find_element(By.XPATH, "..")
            style = parent.get_attribute("style")
            if "visibility: hidden" not in style and "display: none" not in style:
                visible_buttons.append(element)
        
        canvases = []

        for element in visible_buttons:
            try:
                canvas = element.find_element(By.TAG_NAME, "canvas")
                canvases.append(canvas)
            except:
                pass

        return canvases

    def get_board_image(self):
        canvas = self.get_board_canvas()
        image = self.decode_canvas(canvas)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def get_scoreboard_image(self):
        canvas = self.get_scoreboard_canvas()
        image = self.decode_canvas(canvas)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def get_meld_images(self):
        canvases = self.get_meld_canvases()
        images = []
        for canvas in canvases:
            image = self.decode_canvas(canvas)
            images.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return images

    def save_board_image(self, file_name="board"):
        image = self.get_board_image()
        cv2.imwrite(f"{file_name}.png", image)

    def save_scoreboard_image(self, file_name="scoreboard"):
        image = self.get_scoreboard_image()
        cv2.imwrite(f"{file_name}.png", image)

    def save_meld_images(self, file_name="call"):
        images = self.get_meld_images()
        for i, image in enumerate(images):
            cv2.imwrite(f"{file_name}_{i}.png", image)

    def decode_canvas(self, canvas):
        canvas_base64 = self.driver.execute_script(
            "return arguments[0].toDataURL('image/png').substring(21);", canvas)
        canvas_png = base64.b64decode(canvas_base64)
        return imread(io.BytesIO(canvas_png))

    def get_page_source(self):
        return self.driver.page_source

    def do_discard(self, index):
        canvas = self.get_board_canvas()
        ac = ActionChains(self.driver)
        ac.move_to_element_with_offset(canvas,
            36 + 40 * index,
            read_board.expected_height - 46) \
            .click().perform()

    def do_action(self, action):
        try:
            element = self.find_visible_button(action)
            ac = ActionChains(self.driver)
            ac.move_to_element(element).click().perform()
        except:
            logging.warning("Failed to click on element")

    def do_meld_option(self, index):
        canvases = self.get_meld_canvases()
        if index < len(canvases):
            ac = ActionChains(self.driver)
            ac.move_to_element(canvases[index]).click().perform()
        else:
            logging.error("Tried to choose an out of bounds meld option.")

    def find_visible_button(self, action):
        bold = (action == "Ron" or action == "Tsumo")
        bold_str = "/b" if bold else ""
        xpath = f"//div{bold_str}[text()[contains(., '{action}')]]"
        if not self.try_find_element(xpath):
            return None
        elements = self.driver.find_elements(By.XPATH, xpath)
        for element in elements:
            parent_path = "../.." if bold else ".."
            parent = element.find_element(By.XPATH, parent_path)
            style = parent.get_attribute("style")
            if "visibility: hidden" not in style and "display: none" not in style:
                return element
        return None

    def try_find_element(self, xpath):
        try:
            self.driver.find_element(By.XPATH, xpath)
            return True
        except:
            return False

    def find_ok_button(self):
        try:
            element = self.driver.find_element(By.XPATH, "//button[@name='ok']")
            return element
        except:
            return None

    def auto_adjust_size(self):
        while True:
            time.sleep(0.5)
            board = self.get_board_image()
            window_size = self.driver.get_window_size()
            if board.shape[1] < read_board.expected_width * self.size_mult or \
               board.shape[0] < read_board.expected_height * self.size_mult:
                window_size["width"] += 1
                window_size["height"] += 1
            elif board.shape[1] > read_board.expected_width * self.size_mult or \
                 board.shape[0] < read_board.expected_height * self.size_mult:
                window_size["width"] -= 1
            else:
                break
            self.driver.set_window_size(window_size["width"], window_size["height"])

    def get_player_state(self):
        self.board_reader.read(self.get_board_image())
        self.scoreboard_reader.read(self.get_scoreboard_image())
        try:
            player_state = PlayerState(
                self.board_reader.get_private_tiles(),
                self.board_reader.get_discarded_tiles(),
                self.board_reader.get_open_tiles(),
                self.board_reader.get_doras(),
                self.scoreboard_reader.get_round_wind(),
                self.scoreboard_reader.get_round_number(),
                self.scoreboard_reader.get_player_scores(),
                self.scoreboard_reader.get_player_wind()
            )
            player_state.verify()
            return player_state
        except Exception as e:
            logging.error("Failed to read player state")
            traceback.print_exc()
            return None

    def get_turn_player(self):
        self.scoreboard_reader.read(self.get_scoreboard_image())
        return self.scoreboard_reader.get_turn_player()

    def get_round_name(self):
        self.scoreboard_reader.read(self.get_scoreboard_image())
        wind = self.scoreboard_reader.get_round_wind()
        round_number = self.scoreboard_reader.get_round_number()
        wind_name = ["East", "South", "West", "North"][wind]
        return f"{wind_name} {round_number}"

    def get_final_player_scores(self):
        rows = self.driver.find_elements(By.CLASS_NAME, "bbg5")
        players_and_scores = []
        for row in rows:
            player_name = row.text
            player_name = player_name[:player_name.index("\n")]
            player_score = int(row.find_element(By.TAG_NAME, "td").text)
            players_and_scores.append((player_name, player_score))
        return players_and_scores

    def get_meld_options(self):
        self.board_reader.read_meld_options(self.get_meld_images())
        return self.board_reader.get_meld_options()

    def reset_cursor(self):
        canvas = self.get_board_canvas()
        ac = ActionChains(self.driver)
        ac.move_to_element_with_offset(canvas, 6, 6).perform()

    def stop(self):
        self.driver.close()
        self.driver = None

class PlayerState:

    TILES = [
        "1m", "2m", "3m", "4m", 
        "5m", "6m", "7m", "8m", "9m",
        "1p", "2p", "3p", "4p", 
        "5p", "6p", "7p", "8p", "9p", 
        "1s", "2s", "3s", "4s", 
        "5s", "6s", "7s", "8s", "9s", 
        "ew", "sw", "ww", "nw",       
        "wd", "gd", "rd" 
    ]

    WINDS = ["east", "south", "west", "north"]

    def __init__(self, private_tiles, discarded_tiles, open_tiles,
                    dora_indicators, round_wind, round_number,
                    player_scores, player_wind):
        self.private_tiles = private_tiles
        self.discarded_tiles = discarded_tiles
        self.open_tiles = open_tiles
        self.dora_indicators = dora_indicators
        self.round = self.get_round(round_wind, round_number - 1)
        self.player_scores = player_scores
        self.player_wind = player_wind
        self.aka_doras_in_hand = self.get_aka_doras_in_hand()
        self.opponent_riichi = [self.in_riichi(player=i) for i in range(1,4)]

    def get_round(self, round_wind, round_number):
        return round_wind * 4 + round_number

    def get_aka_doras_in_hand(self):
        return len([i for i in self.discarded_tiles[0] if i[:1] == "0"])

    def in_riichi(self, player=0):
        return len([i for i in self.discarded_tiles[player] if i[2:3] == "r"])

    def get_called_tile(self):
        for player_discarded_tiles in self.discarded_tiles:
            for tile in player_discarded_tiles:
                if "c" in tile:
                    return tile[:2]
        return None

    def map_tiles(self, tiles):
        return list(map(lambda x: x[0:2].replace("0", "5"), tiles))

    def __str__(self):
        lines = [
            f"PLAYER STATE {self.WINDS[self.round % 4]} {self.round // 4}",
            f"Private Tiles: {' '.join(self.private_tiles)}",
            f"Discarded Tiles:",
            f"  Player:   {' '.join(self.discarded_tiles[0])}",
            f"  Right:    {' '.join(self.discarded_tiles[1])}",
            f"  Opposite: {' '.join(self.discarded_tiles[2])}",
            f"  Left:     {' '.join(self.discarded_tiles[3])}",
            f"Open Tiles:",
            f"  Player:   {' '.join([str(i) for i in self.open_tiles[0]])}",
            f"  Right:    {' '.join([str(i) for i in self.open_tiles[1]])}",
            f"  Opposite: {' '.join([str(i) for i in self.open_tiles[2]])}",
            f"  Left:     {' '.join([str(i) for i in self.open_tiles[3]])}",
            f"Dora Indicators: {' '.join(self.dora_indicators)}",
            f"Player Scores: {' '.join([str(i) for i in self.player_scores])}",
            f"Player Wind: {self.WINDS[self.player_wind]}",
            f"Aka Doras in Hand: {self.aka_doras_in_hand}",
            f"Riichi Status: {[self.in_riichi(player=i) for i in range(0,4)]}"
        ]
        return '\n'.join(lines)

    def verify(self):
        tile_counts = [0] * 34
        
        for tile in [i[:2].replace("0", "5") for i in self.private_tiles]:
            tile_counts[self.TILES.index(tile)] += 1
        
        for pile in self.discarded_tiles:
            for tile in [i[:2].replace("0", "5") for i in pile]:
                tile_counts[self.TILES.index(tile)] += 1
        
        for pile in self.open_tiles:
            for meld in pile:
                for tile in [i[:2].replace("0", "5") for i in meld]:
                    tile_counts[self.TILES.index(tile)] += 1

        for tile in [i[:2].replace("0", "5") for i in self.dora_indicators]:
            tile_counts[self.TILES.index(tile)] += 1

        for i, count in enumerate(tile_counts):
            if count > 4:
                logging.error(f"There are more than 4 {self.TILES[i]} on the board (Total: {count})")

    def to_feature_list(self):
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
            idx = self.TILES.index(tile)
            for i in range(4):
                if feature_private_tiles[idx + i * 34] == 0:
                    feature_private_tiles[idx + i * 34] = 1
                    break

        # self.discarded_tiles
        feature_discarded_tiles = [0] * 34 * 30 * 4
        for player_i, discarded_tiles in enumerate(self.discarded_tiles):
            mapped_discarded_tiles = self.map_tiles(discarded_tiles)
            for i, tile in enumerate(mapped_discarded_tiles):
                idx = self.TILES.index(tile)
                feature_discarded_tiles[idx + (i * 34) + (player_i * 34 * 30)] = 1

        # self.open_tiles
        feature_open_tiles = [0] * 34 * 4 * 4
        for player_i, player_melds in enumerate(self.open_tiles):
            for i, meld in enumerate(player_melds):
                mapped_meld = self.map_tiles(meld)
                for tile in mapped_meld:
                    idx = self.TILES.index(tile)
                    for j in range(4):
                        if feature_open_tiles[idx + (j * 34 + (player_i * 34 * 4))] == 0:
                            feature_open_tiles[idx + (j * 34 + (player_i * 34 * 4))] = 1
                            break

        # self._dora_indicators
        feature_dora_indicators = [0] * 34 * 4
        mapped_dora_indicators = self.map_tiles(self.dora_indicators)
        for tile in mapped_dora_indicators:
            idx = self.TILES.index(tile)
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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start an automated Chrome browser with Tenhou bot functionality.")
    parser.add_argument("-S", "--sizemult", dest="size_mult", type=int, default=1, help="Apply a modifier to the screen size (default: 1)")
    args = parser.parse_args()

    if args.size_mult < 1:
        print("sizemult must be at least 1")
        sys.exit(-1)

    main(size_mult=args.size_mult)