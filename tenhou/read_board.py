import cv2
import math
import read_tile

private_tile_model = read_tile.create_private_tile_model()
print("Private tile model finished")
discard_tile_model = read_tile.create_discard_tile_model()
#discard_h_tile_model, discard_v_tile_model = read_tile.create_discard_tile_models()
print("Discard tile models finished")

def main():
    board = cv2.imread("board.png", cv2.IMREAD_COLOR)
    board = cv2.resize(board, (600, 545), interpolation=cv2.INTER_LINEAR)

    tiles = get_player_tiles(board)
    print("Player Hand")
    print(' '.join(tiles))

    print("Discared Tiles")
    discarded_tiles = get_discarded(board)
    for i in range(4):
        print(' '.join(discarded_tiles[i]))

    doras = get_doras(board)
    print("Doras:", ' '.join(doras))

def get_player_tiles(board):
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
        b, g, r = board[hand_y, x]
        if g > 190 and g < 200:
            tile_y = hand_y
            while True:
                tile_y -= 1
                if board[tile_y, x][1] > 220:
                    break
            tile_height = hand_y - tile_y
            while True:
                tile_height += 1
                if board[tile_y + tile_height, x][1] == 0:
                    tile_height -= 1
                    break
            break

    # Find the left and right of all tiles in a row
    for x in range(0, board.shape[1]):
        b, g, r = board[tile_y, x]
        if at_tile:
            if g < 190 or g > 230:
                tile_width = x - tile_x
                if tile_width > tile_height / 3:
                    tiles.append((tile_x, tile_y, tile_width, tile_height))
                at_tile = False
        else:
            if g > 190 and g <= 230:
                tile_x = x - 1
                at_tile = True
                drought = 0
            elif len(tiles) > 0:
                # Stop if no further tiles are found in a while (avoid called tiles)
                drought += 1
                if drought > 8:
                    break

    images = []
    for x, y, width, height in tiles:
        tile_img = board[y:(y+height), (x-2):(x+width)]
        tile_img = read_tile.preprocess(tile_img)
        images.append(tile_img)

    return read_tile.predict(private_tile_model, images)

def get_discarded(board, save_img=False):
    # player, right, opposite, left
    discarded = [
        get_discard_pile(board, 227, 312,  25,  28, 21, 24, owner=0, save_img=save_img),
        get_discard_pile(board, 377, 290,  33, -21, 29, 19, owner=1, save_img=save_img),
        get_discard_pile(board, 352, 158, -25, -28, 21, 24, owner=2, save_img=save_img),
        get_discard_pile(board, 194, 185, -33,  21, 29, 19, owner=3, save_img=save_img)
    ]

    return discarded

def get_discard_pile(board, x, y, tile_width, tile_height, icon_width, icon_height, owner=0, save_img=False):
    tile_images = []
    original_tile_images = []
    while True:
        if sum(board[y, x]) < 10 or \
           sum(board[y+icon_height, x]) < 10 or \
           sum(board[y, x + icon_width]) < 10 or \
           sum(board[y+icon_height, x+icon_width]) < 10:
            # Found no full tile
            break
        tile_img = board[y:(y+icon_height), x:(x+icon_width)]
        if owner == 1:
            tile_img = cv2.rotate(tile_img, cv2.ROTATE_90_CLOCKWISE)
        elif owner == 2:
            tile_img = cv2.rotate(tile_img, cv2.ROTATE_180)
        elif owner == 3:
            tile_img = cv2.rotate(tile_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        original_tile_images.append(tile_img)
        tile_img = read_tile.preprocess(tile_img)
        tile_images.append(tile_img)
        if owner % 2 == 0:
            x += tile_width
            if len(tile_images) <= 12 and len(tile_images) % 6 == 0:
                x -= tile_width * 6
                y += tile_height
        else:
            y += tile_height
            if len(tile_images) <= 12 and len(tile_images) % 6 == 0:
                y -= tile_height * 6
                x += tile_width
    
    if len(tile_images) > 0:
        discarded = read_tile.predict(discard_tile_model, tile_images)
        if save_img:
            owner_name = ["player", "right", "opposite", "left"][owner]
            for i, img in enumerate(original_tile_images):
                tile_name = discarded[i]
                file_name = f"data/test/{owner_name}/{tile_name}_{100*(owner+1)+i}.png"
                cv2.imwrite(file_name, img)
        return discarded

    return []

def get_discarded_by_scan(board, save_img=False):
    # player, right, opposite, left
    discarded = []

    # player tiles
    _, top_y = scan_start(board, 146, 220, 146, 240)
    tile_height = 20
    tile_width = 16
    tile_images = []
    original_tile_images = []
    for row in range(3):
        y = round(top_y + tile_height/2 + tile_height*row)
        x, _, width, _ = scan_start_end(board, 130, y, 300, y)
        num_tiles = round(width / tile_width)
        if num_tiles == 0:
            continue
        real_tile_width = width / num_tiles
        for i in range(num_tiles):
            start_x = round(x + real_tile_width * i)
            end_x   = round(x + min([real_tile_width * (i+1), width]))
            start_y = top_y + tile_height * row + 1
            end_y   = top_y + tile_height * (row + 1)
            tile_img = board[start_y:end_y, start_x:end_x]
            original_tile_images.append(tile_img)
            tile_img = read_tile.preprocess(tile_img)
            tile_images.append(tile_img)
    
    if len(tile_images) > 0:
        discarded.append(read_tile.predict(discard_tile_model, tile_images))
        if save_img:
            for i, img in enumerate(original_tile_images):
                tile_name = discarded[0][i]
                file_name = f"data/test/player/{tile_name}_p{300+i}.png"
                cv2.imwrite(file_name, img)
    else:
        discarded.append([])
    
    # right tiles
    left_x, _ = scan_start(board, 230, 224, 260, 224)
    tile_height = 16
    tile_width = 21
    tile_images = []
    original_tile_images = []
    for row in range(3):
        x = round(left_x + tile_width/2 + tile_width*row)
        _, y, _, height = scan_start_end(board, x, 76, x, 260)
        height -= 6 # side of tile
        num_tiles = round(height / tile_height)
        if num_tiles == 0:
            continue
        real_tile_height = height / num_tiles
        for j in range(num_tiles):
            i = num_tiles - j - 1
            start_y = round(y + real_tile_height * i)
            end_y   = round(y + min([real_tile_height * (i+1), height]))
            start_x = left_x + tile_width * row + 1
            end_x   = left_x + tile_width * (row + 1)
            tile_img = board[start_y:end_y, start_x:end_x]
            tile_img = cv2.rotate(tile_img, cv2.ROTATE_90_CLOCKWISE)
            original_tile_images.append(tile_img)
            tile_img = read_tile.preprocess(tile_img)
            tile_images.append(tile_img)
    
    if len(tile_images) > 0:
        discarded.append(read_tile.predict(discard_tile_model, tile_images))
        if save_img:
            for i, img in enumerate(original_tile_images):
                tile_name = discarded[1][i]
                file_name = f"data/test/right/{tile_name}_r{300+i}.png"
                cv2.imwrite(file_name, img)
    else:
        discarded.append([])
    
    # opposite tiles
    _, bottom_y = scan_start(board, 224, 140, 224, 100)
    bottom_y -= 5 # side of tile
    tile_height = 20
    tile_width = 16
    tile_images = []
    original_tile_images = []
    for row in range(3):
        y = round(bottom_y - tile_height/2 - tile_height*row)
        x, _, width, _ = scan_start_end(board, 68, y, 240, y)
        num_tiles = round(width / tile_width)
        if num_tiles == 0:
            continue
        real_tile_width = width / num_tiles
        for j in range(num_tiles):
            i = num_tiles - j - 1
            start_x = round(x + real_tile_width * i)
            end_x   = round(x + min([real_tile_width * (i+1), width]))
            start_y = bottom_y - tile_height * (row + 1) + 1
            end_y   = bottom_y - tile_height * row
            tile_img = board[start_y:end_y, start_x:end_x]
            tile_img = cv2.rotate(tile_img, cv2.ROTATE_180)
            original_tile_images.append(tile_img)
            tile_img = read_tile.preprocess(tile_img)
            tile_images.append(tile_img)
    
    if len(tile_images) > 0:
        discarded.append(read_tile.predict(discard_tile_model, tile_images))
        if save_img:
            for i, img in enumerate(original_tile_images):
                tile_name = discarded[2][i]
                file_name = f"data/test/opposite/{tile_name}_o{300+i}.png"
                cv2.imwrite(file_name, img)
    else:
        discarded.append([])
    
    # left tiles
    right_x, _ = scan_start(board, 150, 146, 110, 146)
    tile_height = 16
    tile_width = 21
    tile_images = []
    original_tile_images = []
    for row in range(3):
        x = round(right_x - tile_width/2 - tile_width*row)
        _, y, _, height = scan_start_end(board, x, 76, x, 260)
        height -= 6 # side of tile
        num_tiles = round(height / tile_height)
        if num_tiles == 0:
            continue
        real_tile_height = height / num_tiles
        for i in range(num_tiles):
            start_y = round(y + real_tile_height * i)
            end_y   = round(y + min([real_tile_height * (i+1), height]))
            start_x = right_x - tile_width * (row + 1) + 1
            end_x   = right_x - tile_width * row + 1
            tile_img = board[start_y:end_y, start_x:end_x]
            tile_img = cv2.rotate(tile_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            original_tile_images.append(tile_img)
            tile_img = read_tile.preprocess(tile_img)
            tile_images.append(tile_img)
    
    if len(tile_images) > 0:
        discarded.append(read_tile.predict(discard_tile_model, tile_images))
        if save_img:
            for i, img in enumerate(original_tile_images):
                tile_name = discarded[3][i]
                file_name = f"data/test/left/{tile_name}_l{300+i}.png"
                cv2.imwrite(file_name, img)
    else:
        discarded.append([])

    return discarded

def get_doras(board):
    icon_width_v = 21
    icon_height_v = 24
    icon_width_h = 29
    icon_height_v = 19
    #get_discard_pile(board, 227, 312,  25,  28, 21, 24, owner=0, save_img=save_img),
    #get_discard_pile(board, 377, 290,  33, -21, 29, 19, owner=1, save_img=save_img),

def get_wall_doras(board, owner=0):
    x = []

def scan_start(board, from_x, from_y, to_x, to_y, min_value=0):
    while from_x != to_x or from_y != to_y:
        if board[from_y, from_x] > min_value:
            return from_x, from_y
        if from_x < to_x:
            from_x += 1
        elif from_x > to_x:
            from_x -= 1
        if from_y < to_y:
            from_y += 1
        elif from_y > to_y:
            from_y -= 1

    return 0, 0 # Found no tiles

def scan_start_end(board, from_x, from_y, to_x, to_y, min_value=0, max_drought=5):
    drought = 0
    length_x = 0
    length_y = 0

    from_x, from_y = scan_start(board, from_x, from_y, to_x, to_y, min_value=min_value)

    while from_x + length_x != to_x or from_y + length_y != to_y:
        if board[from_y + length_y, from_x + length_x] <= min_value:
            drought += 1
            if drought > max_drought:
                length_x = abs(length_x)
                length_y = abs(length_y)
                if from_x != to_x:
                    length_x -= max_drought
                if from_y != to_y:
                    length_y -= max_drought
                return from_x, from_y, length_x, length_y
        else:
            drought = 0
        if from_x + length_x < to_x:
            length_x += 1
        elif from_x + length_x > to_x:
            length_x -= 1
        if from_y + length_y < to_y:
            length_y += 1
        elif from_y + length_y > to_y:
            length_y -= 1

    return 0, 0, 0, 0 # Found no tiles


if __name__ == "__main__":
    main()