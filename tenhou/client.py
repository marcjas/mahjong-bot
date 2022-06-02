from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import base64
import cv2
from imageio.v2 import imread
import io
import read_board
import sys

tenhou_id = "ID57255B6D-m3eZdhEG"
tenhou_url = "https://tenhou.net/3/"

def main():
    options = webdriver.ChromeOptions()
    options.add_argument("--allow-file-access-from-files")
    options.add_argument("--disable-web-security")
    options.add_argument("--log-level=3")
    options.add_argument("--disable-infobars")
    options.add_experimental_option("useAutomationExtension", False)
    options.add_experimental_option("excludeSwitches", ["enable-automation"])

    driver = webdriver.Chrome(options=options)
    driver.set_window_size(616, 700)
    driver.get(tenhou_url)
    time.sleep(5)

    while True:
        command = input("Command: ").lower()
        
        if command == "help" or command == "?":
            print("help: This menu")
            print("?:    This menu")
            print("read: Read the board")
            print("stop: Exit the program")
        elif command == "read":
            read(driver)
        elif command == "save":
            save(driver)
        elif command == "stop":
            driver.close()
            break

def read(driver):
    canvases = driver.find_elements(By.TAG_NAME, "canvas")

    board = decode_canvas(driver, canvases[1])
    board = cv2.cvtColor(board, cv2.COLOR_RGB2BGR)
    if board.shape[0] != 545 and board[1] != 600:
        print("Window size was changed. Please restart the program")
        #board = cv2.resize(board, (600, 545), interpolation=cv2.INTER_LINEAR)
        driver.close()
        sys.exit()
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

def decode_canvas(driver, canvas):
    canvas_base64 = driver.execute_script("return arguments[0].toDataURL('image/png').substring(21);", canvas)
    canvas_png = base64.b64decode(canvas_base64)
    return imread(io.BytesIO(canvas_png))

if __name__ == "__main__":
    main()