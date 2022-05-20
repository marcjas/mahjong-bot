from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import base64
import cv2
from imageio.v2 import imread
import io
import read_board

tenhou_id = "ID57255B6D-m3eZdhEG"
tenhou_url = "https://tenhou.net/3/"

def main():
    options = webdriver.ChromeOptions()
    options.add_argument("--allow-file-access-from-files")
    options.add_argument("--disable-web-security")

    driver = webdriver.Chrome(options=options)
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
    board = cv2.cvtColor(board, cv2.COLOR_RGB2GRAY)
    player_tiles = read_board.get_player_tiles(board)
    print(' '.join(player_tiles))
    #cv2.imwrite("tiles.png", cv2_img)

    #for i, canvas in enumerate(canvases):
    #    driver.execute_script("arguments[0].crossOrigin = 'anonymous';", canvas)
    #    time.sleep(1)
    #    canvas_base64 = driver.execute_script("return arguments[0].toDataURL('image/png').substring(21);", canvas)
    #    canvas_png = base64.b64decode(canvas_base64)

    #    with open(f"canvas_{i}.png", 'wb') as f:
    #        f.write(canvas_png)

def save(driver):
    canvases = driver.find_elements(By.TAG_NAME, "canvas")

    board = decode_canvas(driver, canvases[1])
    board = cv2.cvtColor(board, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("board.png", board)

def decode_canvas(driver, canvas):
    canvas_base64 = driver.execute_script("return arguments[0].toDataURL('image/png').substring(21);", canvas)
    canvas_png = base64.b64decode(canvas_base64)
    return imread(io.BytesIO(canvas_png))

if __name__ == "__main__":
    main()