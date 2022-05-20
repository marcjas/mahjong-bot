import cv2
import math
import read_tile

tile_model = read_tile.create_model()

def main():
    board = cv2.imread("board_call3.png", cv2.IMREAD_GRAYSCALE)

    tiles = get_player_tiles(board)
    print(' '.join(tiles))

def get_player_tiles(board):
    board = cv2.resize(board, (370, 400), interpolation=cv2.INTER_LINEAR)

    hand_y = math.floor(board.shape[0] * 0.915)
    
    tiles = []
    tile_x = 0
    tile_y = 0
    tile_width = 0
    tile_height = 0
    at_tile = False
    drought = 0

    # Find top and bottom of the left-most tile
    for x in range(0, board.shape[1]):
        pixel = board[hand_y, x]
        if pixel > 190 and pixel < 200:
            tile_y = hand_y
            while True:
                tile_y -= 1
                if board[tile_y, x] > 220:
                    break
            tile_height = hand_y - tile_y
            while True:
                tile_height += 1
                if board[tile_y + tile_height, x] == 0:
                    tile_height -= 1
                    break
            break

    # Find the left and right of all tiles in a row
    for x in range(0, board.shape[1]):
        pixel = board[tile_y, x]
        if at_tile:
            if pixel < 190 or pixel > 230:
                tile_width = x - tile_x
                if tile_width > tile_height / 3:
                    tiles.append((tile_x, tile_y, tile_width, tile_height))
                at_tile = False
        else:
            if pixel > 190 and pixel <= 230:
                tile_x = x - 1
                at_tile = True
                drought = 0
            elif len(tiles) > 0:
                # Stop if no further tiles are found in a while (avoid called tiles)
                drought += 1
                if drought > 5:
                    break

    images = []
    for x, y, width, height in tiles:
        tile_img = board[y:(y+height), x:(x+width)]
        tile_img = cv2.resize(tile_img, (read_tile.in_width, read_tile.in_height))
        images.append(tile_img)

    return read_tile.predict(tile_model, images)

if __name__ == "__main__":
    main()