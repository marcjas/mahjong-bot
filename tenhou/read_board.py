import cv2
import math
import read_tile
import sys

expected_width  = 600
expected_height = 545

private_tile_model = read_tile.create_private_tile_model()
print("Private tile model finished")
discard_tile_model = read_tile.create_discard_tile_model()
#discard_h_tile_model, discard_v_tile_model = read_tile.create_discard_tile_models()
print("Discard tile models finished")

def main(filename):
    board = cv2.imread(f"boards/{filename}.png", cv2.IMREAD_COLOR)
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

    open_tiles = get_open_tiles(board)
    for i in range(4):
        print(open_tiles[i])

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
                if drought > 16:
                    break

    images = []
    for x, y, width, height in tiles:
        tile_img = board[y:(y+height), (x-2):(x+width)]
        tile_img = read_tile.preprocess(tile_img)
        images.append(tile_img)

    return read_tile.predict(private_tile_model, images)

def get_open_tiles(board, save_img=False):
    open_tiles = [
        get_open_tile_pile(board, 577, 488, owner=0, save_img=save_img),
        get_open_tile_pile(board, 569, 65, owner=1, save_img=save_img),
        get_open_tile_pile(board, 73, 1, owner=2, save_img=save_img),
        get_open_tile_pile(board, 2, 373, owner=3, save_img=save_img)
    ]

    return open_tiles

def get_open_tile_pile(board, x, y, owner=0, save_img=False):
    tile_widths  = [25,  33, -25, -33]
    tile_heights = [28, -21, -28,  21]
    icon_widths  = [21,  29]
    icon_heights = [24,  19]

    tile_width  = tile_widths[owner]
    tile_height = tile_heights[owner]
    icon_width  = icon_widths[owner % 2]
    icon_height = icon_heights[owner % 2]
    kan_width   = [0, -18, 0, 47][owner]
    kan_height  = [-14, 16, 44, 16][owner]

    open_sets = []
    sideways_index = -1

    tile_images = []
    set_index = 0
    closed_kan_index = 0
    while True:
        sideways = False
        kan_tile = (sum(board[y+kan_height, x+kan_width]) > 200)
        if (sum(board[y+icon_height-2, x+1]) > 10 and \
            sum(board[y+icon_height-2, x+1]) < 100 and \
            sum(board[y+icon_height-2, x+icon_width-2]) > 10 and \
            sum(board[y+icon_height-2, x+icon_width-2]) < 100) or \
            closed_kan_index == 3:
            # Found the start of a closed kan
            if closed_kan_index == 0:
                closed_kan_index = 1
        elif sum(board[y, x]) < 440 or \
           sum(board[y+icon_height-1, x]) < 440 or \
           sum(board[y, x+icon_width-1]) < 440 or \
           sum(board[y+icon_height-1, x+icon_width-1]) < 440 or \
           kan_tile:
            # Found no regular tile
            tx = x
            ty = y
            if owner == 0:
                ty += (tile_height - tile_heights[3]) - 1
                tx -= (tile_widths[1] - tile_width)
            elif owner == 1:
                tx += (tile_width - tile_widths[0])
                ty += 1
            elif owner == 2:
                ty -= 1
            elif owner == 3:
                ty -= (tile_heights[0] - tile_height) - 1
            ticon_width  = icon_widths[(owner + 1) % 2]
            ticon_height = icon_heights[(owner + 1) % 2]
            if sum(board[ty, tx]) < 440 or \
               sum(board[ty+ticon_height-1, tx]) < 440 or \
               sum(board[ty, tx + ticon_width-1]) < 440 or \
               sum(board[ty+ticon_height-1, tx+ticon_width-1]) < 440:
                    # Found no sideways tile
                    tile_img = board[ty:(ty+ticon_height), tx:(tx+ticon_width)]
                    #cv2.imwrite("Fail.png", tile_img)
                    break
            else:
                # Sideways tile
                sideways = True
                sideways_index = set_index
                tile_img = board[ty:(ty+ticon_height), tx:(tx+ticon_width)]
                tile_img = cv2.rotate(tile_img, cv2.ROTATE_90_CLOCKWISE)
                if closed_kan_index == 1:
                    closed_kan_index = 2
        else:
            # Regular tile
            tile_img = board[y:(y+icon_height), x:(x+icon_width)]
        # Processing
        if owner % 2 == 0:
            x -= tile_widths[(owner + 1) % 4] if sideways else tile_width
        else:
            y -= tile_heights[(owner + 1) % 4] if sideways else tile_height
        if closed_kan_index > 0:
            pass
            #print(closed_kan_index, owner, set_index, len(open_sets))
        if closed_kan_index == 1:
            continue
        if closed_kan_index != 3:
            if owner == 1:
                tile_img = cv2.rotate(tile_img, cv2.ROTATE_90_CLOCKWISE)
            elif owner == 2:
                tile_img = cv2.rotate(tile_img, cv2.ROTATE_180)
            elif owner == 3:
                tile_img = cv2.rotate(tile_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if save_img:
                cv2.imwrite(f"{owner}_{len(open_sets)}_{len(tile_images)}.png", tile_img)
            tile_img = read_tile.preprocess(tile_img)
            tile_images.append(tile_img)
            if kan_tile:
                tile_images.append(tile_img)
                if closed_kan_index == 2:
                    set_index = 2
                    closed_kan_index = 3
                    tile_images.append(tile_img)
                    tile_images.append(tile_img)
                    continue
        set_index += 1
        if set_index == 3:
            open_set = read_tile.predict(discard_tile_model, tile_images)
            if not closed_kan_index == 3:
                open_set[sideways_index] += ["p", "r", "o", "l"][(owner - 3 + sideways_index) % 4]
            else:
                closed_kan_index = 0
            if len(open_set) == 4:
                for i, tile in enumerate(open_set):
                    if tile[0:1] == "0" or tile[0:1] == "5":
                        open_set[i] = ("0" if i == 0 else "5") + tile[1:]
            open_sets.append(open_set)
            set_index = 0
            tile_images = []


    return open_sets

def get_discarded(board, save_img=False):
    # player, right, opposite, left
    discarded = [
        get_discard_pile(board, 227, 312, owner=0, save_img=save_img),
        get_discard_pile(board, 377, 290, owner=1, save_img=save_img),
        get_discard_pile(board, 352, 158, owner=2, save_img=save_img),
        get_discard_pile(board, 194, 185, owner=3, save_img=save_img)
    ]

    return discarded

def get_discard_pile(board, x, y, owner=0, save_img=False):
    tile_widths  = [25,  33, -25, -33]
    tile_heights = [28, -21, -28,  21]
    icon_widths  = [21,  29]
    icon_heights = [24,  19]

    tile_width  = tile_widths[owner]
    tile_height = tile_heights[owner]
    icon_width  = icon_widths[owner % 2]
    icon_height = icon_heights[owner % 2]

    base_x = x
    base_y = y
    riichi_index = -1
    calling_tile = False

    tile_images = []
    original_tile_images = []
    while True:
        riichi = False
        tile_img = None
        if sum(board[y, x]) < 440 or \
           sum(board[y+icon_height-1, x]) < 440 or \
           sum(board[y, x+icon_width-1]) < 440 or \
           sum(board[y+icon_height-1, x+icon_width-1]) < 440:
            # Found no regular tile
            rx = x
            ry = y
            if owner == 0:
                ry += (tile_height - tile_heights[3]) - 1
            elif owner == 1:
                rx += (tile_width - tile_widths[0])
            elif owner == 2:
                rx -= (tile_width - tile_widths[3])
                ry -= 1
            ricon_width  = icon_widths[(owner + 1) % 2]
            ricon_height = icon_heights[(owner + 1) % 2]
            if sum(board[ry, rx]) < 440 or \
               sum(board[ry+ricon_height-1, rx]) < 440 or \
               sum(board[ry, rx + ricon_width-1]) < 440 or \
               sum(board[ry+ricon_height-1, rx+ricon_width-1]) < 440:
                    # Found no riichi tile
                    cx = x + [5, 4, -5, -4][owner]
                    cy = y + [4, -4, -4, 4][owner]
                    if sum(board[cy, cx]) < 440 or \
                       sum(board[cy+icon_height-1, cx]) < 440 or \
                       sum(board[cy, cx + icon_width-1]) < 440 or \
                       sum(board[cy+icon_height-1, cx+icon_width-1]) < 440:
                        # Found no regular tile ready to be called
                        rx = rx + [5, 4, -5, -4][owner]
                        ry = ry + [4, -4, -4, 4][owner]
                        if sum(board[ry, rx]) < 440 or \
                           sum(board[ry+ricon_height-1, rx]) < 440 or \
                           sum(board[ry, rx + ricon_width-1]) < 440 or \
                           sum(board[ry+ricon_height-1, rx+ricon_width-1]) < 440:
                            # Found no riichi tile ready to be called
                            break
                        else:
                            riichi = True
                            if riichi_index >= 0:
                                print("Found two riichi tiles, this is impossible.")
                                continue
                            riichi_index = len(tile_images)
                            tile_img = board[ry:(ry+ricon_height), rx:(rx+ricon_width)]
                            tile_img = cv2.rotate(tile_img, cv2.ROTATE_90_CLOCKWISE)
                    else:
                        calling_tile = True
                        tile_img = board[cy:(cy+icon_height), cx:(cx+icon_width)]
            else:
                # Riichii tile
                riichi = True
                if riichi_index >= 0:
                    print("Found two riichi tiles, this is impossible.")
                    continue
                riichi_index = len(tile_images)
                tile_img = board[ry:(ry+ricon_height), rx:(rx+ricon_width)]
                tile_img = cv2.rotate(tile_img, cv2.ROTATE_90_CLOCKWISE)
        else:
            # Regular tile
            tile_img = board[y:(y+icon_height), x:(x+icon_width)]
        # Processing
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
            x += tile_widths[(owner + 1) % 4] if riichi else tile_width
            if len(tile_images) <= 12 and len(tile_images) % 6 == 0:
                x = base_x
                y += tile_height
        else:
            y += tile_heights[(owner + 1) % 4] if riichi else tile_height
            if len(tile_images) <= 12 and len(tile_images) % 6 == 0:
                y = base_y
                x += tile_width
    
    if len(tile_images) > 0:
        discarded = read_tile.predict(discard_tile_model, tile_images)
        if save_img:
            owner_name = ["player", "right", "opposite", "left"][owner]
            for i, img in enumerate(original_tile_images):
                tile_name = discarded[i]
                file_name = f"data/test/{owner_name}/{tile_name}_{100*(owner+1)+i}.png"
                cv2.imwrite(file_name, img)
        if riichi_index >= 0:
            discarded[riichi_index] += "r"
        if calling_tile:
            discarded[len(discarded)-1] += "c"
        return discarded

    return []

def get_doras(board):
    doras = []
    doras += get_wall_doras(board, 0)
    doras += get_wall_doras(board, 1)
    doras += get_wall_doras(board, 2)
    doras += get_wall_doras(board, 3)
    return doras

def get_wall_doras(board, owner=0):
    icon_width = 21 if owner % 2 == 0 else 29
    icon_height = 24 if owner % 2 == 0 else 19
    tile_width = 25 if owner % 2 == 0 else 33
    tile_height = 28 if owner % 2 == 0 else 21
    x = [104, 498, 475, 73][owner]
    y = [409, 389, 52, 77][owner]

    doras = []

    while True:
        if sum(board[y, x]) < 140 or \
           sum(board[y+icon_height, x]) < 140 or \
           sum(board[y, x + icon_width]) < 140 or \
           sum(board[y+icon_height, x+icon_width]) < 140:
            # Found no tile
            match owner:
                case 0:
                    x += 1
                    if x > 510:
                        break
                case 1:
                    y -= 1
                    if y < 60:
                        break
                case 2:
                    x -= 1
                    if x < 70:
                        break
                case 3:
                    y += 1
                    if y > 440:
                        break
        else:
            # Found a tile
            if owner == 0:
                x += 2
            elif owner == 1:
                y -= 1
            elif owner == 2:
                x -= 1
            tile_img = board[y:(y+icon_height), x:(x+icon_width)]
            match owner:
                case 0:
                    x += tile_width
                case 1:
                    tile_img = cv2.rotate(tile_img, cv2.ROTATE_90_CLOCKWISE)
                    y -= tile_height
                case 2:
                    tile_img = cv2.rotate(tile_img, cv2.ROTATE_180)
                    x -= tile_width
                case 3:
                    tile_img = cv2.rotate(tile_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    y += tile_height
            tile_img = read_tile.preprocess(tile_img)

            doras.append(tile_img)

    if len(doras) == 0:
        return []

    return read_tile.predict(discard_tile_model, doras)

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
    if len(sys.argv) != 2:
        print("Needs to receive the name of a file from the boards directory.")
        print("The filename should be without the directory and extension.")
        print("Usage: python tenhou/read_board.py <filename>")
        sys.exit(-1)
    main(sys.argv[1])