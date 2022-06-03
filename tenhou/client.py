from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import base64
import cv2
from imageio.v2 import imread
import io
import read_board
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
    driver.set_window_size(read_board.expected_width * size_mult + 16, read_board.expected_height * size_mult + 160)
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
            print("help: This menu")
            print("?:    This menu")
            print("read: Read the board")
            print("stop: Exit the program")
        elif command == "read":
            read(driver, size_mult)
        elif command == "adjust":
            auto_adjust(driver, size_mult)
        elif command == "save":
            save(driver)
        elif command == "stop":
            driver.close()
            break

def read(driver, size_mult=1):
    canvases = driver.find_elements(By.TAG_NAME, "canvas")
    board = decode_canvas(driver, canvases[1])
    board = cv2.cvtColor(board, cv2.COLOR_RGB2BGR)

    if board.shape[1] != read_board.expected_width * size_mult:
        print(f"Window size was changed. Auto-adjusting...")
        auto_adjust(driver, size_mult)
        canvases = driver.find_elements(By.TAG_NAME, "canvas")
        board = decode_canvas(driver, canvases[1])
        board = cv2.cvtColor(board, cv2.COLOR_RGB2BGR)

    if board.shape[1] != read_board.expected_width:
        board = cv2.resize(board, (read_board.expected_width, read_board.expected_height))
    
    player_tiles = read_board.get_player_tiles(board)
    print(' '.join(player_tiles))

    discarded_tiles = read_board.get_discarded(board)
    print("Discarded:")
    for i in range(4):
        print(' '.join(discarded_tiles[i]))

    doras = read_board.get_doras(board)
    print("Doras:", ' '.join(doras))

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