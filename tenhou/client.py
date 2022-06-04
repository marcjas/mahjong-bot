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
            pass
        elif command == "stop":
            driver.close()
            break

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
            choice = input(f"Chii {tile}? (y/n): ")
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