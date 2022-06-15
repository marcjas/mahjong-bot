import cv2
import math
from read_tile import TileModel
import sys

expected_width  = 600
expected_height = 545

def main(filename):
    board = cv2.imread(f"tenhou/examples/boards/{filename}.png", cv2.IMREAD_COLOR)
    board = cv2.resize(board, (600, 545), interpolation=cv2.INTER_LINEAR)

    board_reader = BoardReader(board=board)

    tiles = board_reader.get_private_tiles()
    print("Player Hand")
    print(' '.join(tiles))

    print("Discared Tiles")
    discarded_tiles = board_reader.get_discarded_tiles()
    for i in range(4):
        print(' '.join(discarded_tiles[i]))

    doras = board_reader.get_doras()
    print("Doras:", ' '.join(doras))

    print("Open Tiles:")
    open_tiles = board_reader.get_open_tiles()
    for i in range(4):
        print(open_tiles[i])

class BoardReader:
    def __init__(self, board=None):
        self.private_tile_model = TileModel("models/private_tile_model.pt")
        self.discard_tile_model = TileModel("models/discard_tile_model.pt")
        if board is not None:
            self.read(board)
        else:
            self.board = None
        self.meld_options = None

    def read(self, board):
        board = cv2.resize(board, (expected_width, expected_height))
        self.board = board

    def read_meld_options(self, melds):
        self.meld_options = melds

    def get_meld_options(self):
        melds = []

        for meld in self.meld_options:
            tiles = []
            tile_count = 2
            if meld.shape[0] >= (meld.shape[1] * 1.5):
                tile_count = 1 # Kan
            tile_width = meld.shape[1] // tile_count
            tile_height = (meld.shape[0] * 3) // 4
            for i in range(tile_count):
                tiles.append(meld[
                    (meld.shape[0] - tile_height):(meld.shape[0]-1),
                    (tile_width * i + 1):(tile_width * (i + 1) - 1)])
            tiles = [self.private_tile_model.preprocess(t) for t in tiles]
            if tile_count == 1:
                tiles.append(tiles[0])
                tiles.append(tiles[0])
            melds.append(self.private_tile_model.predict(tiles))

        return melds

    def get_private_tiles(self):
        hand_y = math.floor(self.board.shape[0] * 0.915)

        tiles = []
        tile_x = 0
        tile_y = 0
        tile_width = 0
        tile_height = 0
        at_tile = False
        drought = 0

        # Find top and bottom of the left-most tile
        for x in range(0, self.board.shape[1]):
            b, g, r = self.board[hand_y, x]
            if g > 190 and g < 200:
                tile_y = hand_y
                while True:
                    tile_y -= 1
                    if self.board[tile_y, x][1] > 220:
                        break
                tile_height = hand_y - tile_y
                while True:
                    tile_height += 1
                    if self.board[tile_y + tile_height, x][1] == 0:
                        tile_height -= 1
                        break
                break

        # Find the left and right of all tiles in a row
        for x in range(0, self.board.shape[1]):
            b, g, r = self.board[tile_y, x]
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
            tile_img = self.board[y:(y+height), (x-2):(x+width)]
            tile_img = self.private_tile_model.preprocess(tile_img)
            images.append(tile_img)

        return self.private_tile_model.predict(images)

    def get_open_tiles(self, save_img=False):
        open_tiles = [
            self.get_player_open_tiles(player=0, save_img=save_img),
            self.get_player_open_tiles(player=1, save_img=save_img),
            self.get_player_open_tiles(player=2, save_img=save_img),
            self.get_player_open_tiles(player=3, save_img=save_img)
        ]

        return open_tiles

    def get_player_open_tiles(self, player=0, save_img=False):
        x = [577, 569, 73, 2][player]
        y = [488, 65, 1, 373][player]

        tile_widths  = [25,  33, -25, -33]
        tile_heights = [28, -21, -28,  21]
        icon_widths  = [21,  29]
        icon_heights = [24,  19]

        tile_width  = tile_widths[player]
        tile_height = tile_heights[player]
        icon_width  = icon_widths[player % 2]
        icon_height = icon_heights[player % 2]
        kan_width   = [0, -18, 0, 47][player]
        kan_height  = [-14, 16, 44, 16][player]

        open_sets = []
        sideways_index = -1

        tile_images = []
        set_index = 0
        closed_kan_index = 0
        while True:
            sideways = False
            kan_tile = (sum(self.board[y+kan_height, x+kan_width]) > 200)
            if (sum(self.board[y+icon_height-2, x+1]) > 10 and \
                sum(self.board[y+icon_height-2, x+1]) < 100 and \
                sum(self.board[y+icon_height-2, x+icon_width-2]) > 10 and \
                sum(self.board[y+icon_height-2, x+icon_width-2]) < 100) or \
                closed_kan_index == 3:
                # Found the start of a closed kan
                if closed_kan_index == 0:
                    closed_kan_index = 1
            elif sum(self.board[y, x]) < 440 or \
                sum(self.board[y+icon_height-1, x]) < 440 or \
                sum(self.board[y, x+icon_width-1]) < 440 or \
                sum(self.board[y+icon_height-1, x+icon_width-1]) < 440 or \
                kan_tile:
                # Found no regular tile
                tx = x
                ty = y
                if player == 0:
                    ty += (tile_height - tile_heights[3]) - 1
                    tx -= (tile_widths[1] - tile_width)
                elif player == 1:
                    tx += (tile_width - tile_widths[0])
                    ty += 1
                elif player == 2:
                    ty -= 1
                elif player == 3:
                    ty -= (tile_heights[0] - tile_height) - 1
                ticon_width  = icon_widths[(player + 1) % 2]
                ticon_height = icon_heights[(player + 1) % 2]
                if sum(self.board[ty, tx]) < 440 or \
                    sum(self.board[ty+ticon_height-1, tx]) < 440 or \
                    sum(self.board[ty, tx + ticon_width-1]) < 440 or \
                    sum(self.board[ty+ticon_height-1, tx+ticon_width-1]) < 440:
                        # Found no sideways tile
                        tile_img = self.board[ty:(ty+ticon_height), tx:(tx+ticon_width)]
                        #cv2.imwrite("Fail.png", tile_img)
                        break
                else:
                    # Sideways tile
                    sideways = True
                    sideways_index = set_index
                    tile_img = self.board[ty:(ty+ticon_height), tx:(tx+ticon_width)]
                    tile_img = cv2.rotate(tile_img, cv2.ROTATE_90_CLOCKWISE)
                    if closed_kan_index == 1:
                        closed_kan_index = 2
            else:
                # Regular tile
                tile_img = self.board[y:(y+icon_height), x:(x+icon_width)]
            # Processing
            if player % 2 == 0:
                x -= tile_widths[(player + 1) % 4] if sideways else tile_width
            else:
                y -= tile_heights[(player + 1) % 4] if sideways else tile_height
            if closed_kan_index > 0:
                pass
                #print(closed_kan_index, player, set_index, len(open_sets))
            if closed_kan_index == 1:
                continue
            if closed_kan_index != 3:
                if player == 1:
                    tile_img = cv2.rotate(tile_img, cv2.ROTATE_90_CLOCKWISE)
                elif player == 2:
                    tile_img = cv2.rotate(tile_img, cv2.ROTATE_180)
                elif player == 3:
                    tile_img = cv2.rotate(tile_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                if save_img:
                    cv2.imwrite(f"{player}_{len(open_sets)}_{len(tile_images)}.png", tile_img)
                tile_img = self.discard_tile_model.preprocess(tile_img)
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
                open_set = self.discard_tile_model.predict(tile_images)
                if not closed_kan_index == 3:
                    open_set[sideways_index] += ["p", "r", "o", "l"][(player - 3 + sideways_index) % 4]
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

    def get_discarded_tiles(self, save_img=False):
        # player, right, opposite, left
        discarded = [
            self.get_player_discarded_tiles(player=0, save_img=save_img),
            self.get_player_discarded_tiles(player=1, save_img=save_img),
            self.get_player_discarded_tiles(player=2, save_img=save_img),
            self.get_player_discarded_tiles(player=3, save_img=save_img)
        ]

        return discarded

    def get_player_discarded_tiles(self, player=0, save_img=False):
        x = [227, 377, 352, 194][player]
        y = [312, 290, 158, 185][player]

        player_rotation = [
            None,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_180,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        ][player]

        tile_widths  = [25,  33, -25, -33]
        tile_heights = [28, -21, -28,  21]
        icon_widths  = [21,  29]
        icon_heights = [24,  19]

        tile_width  = tile_widths[player]
        tile_height = tile_heights[player]
        icon_width  = icon_widths[player % 2]
        icon_height = icon_heights[player % 2]

        base_x = x
        base_y = y
        riichi_index = -1
        calling_tile = False

        tile_images = []
        original_tile_images = []
        while True:
            positions = [
                ( # Normal tile
                    x,
                    y,
                    icon_width,
                    icon_height
                ),
                ( # Riichi tile
                    x + [0, (tile_width - tile_widths[0]), -(tile_width - tile_widths[3]), 0][player],
                    y + [(tile_height - tile_heights[3] - 1), -(tile_height - tile_heights[2]), 0, 0][player],
                    icon_widths[(player + 1) % 2],
                    icon_heights[(player + 1) % 2]
                )
            ]
            # Add positions with offsets for called tiles
            called_positions = []
            for pos_x, pos_y, pos_icon_width, pos_icon_height in positions:
                called_positions.append((
                    pos_x + [5, 4, -5, -4][player],
                    pos_y + [4, -4, -4, 4][player],
                    pos_icon_width,
                    pos_icon_height
                ))
            positions += called_positions
            # Check for a valid tile in all four possible positionings
            found_tile = False
            for i, (pos_x, pos_y, pos_icon_width, pos_icon_height) in enumerate(positions):
                pos_right = pos_x + pos_icon_width
                pos_bottom = pos_y + pos_icon_height
                if sum(self.board[pos_y, pos_x]) >= 500 and \
                   sum(self.board[pos_bottom-1, pos_x]) >= 500 and \
                   sum(self.board[pos_y, pos_right-1]) >= 500 and \
                   sum(self.board[pos_bottom-1, pos_right-1]) >= 500:
                    tile_img = self.board[pos_y:pos_bottom, pos_x:pos_right]
                    if i % 2 == 1: # Riichi Tile
                        if riichi_index >= 0:
                            print("Found two riichi tiles, this is impossible.")
                        else:
                            riichi_index = len(tile_images)
                        tile_img = cv2.rotate(tile_img, cv2.ROTATE_90_CLOCKWISE)
                        # Move to next tile
                        if player % 2 == 0:
                            x += tile_widths[(player + 1) % 4]
                        else:
                            y += tile_heights[(player + 1) % 4]
                    else:
                        # Move to next tile
                        if player % 2 == 0:
                            x += tile_width
                        else:
                            y += tile_height
                    # Move down a row
                    if len(tile_images) <= 12 and len(tile_images) % 6 == 5:
                        if player % 2 == 0:
                            x = base_x
                            y += tile_height
                        else:
                            y = base_y
                            x += tile_width
                    if i >= 2: # Calling Tile
                        calling_tile = True
                    if player_rotation is not None: # Rotate to right-side up
                        tile_img = cv2.rotate(tile_img, player_rotation)
                    original_tile_images.append(tile_img)
                    tile_img = self.discard_tile_model.preprocess(tile_img)
                    tile_images.append(tile_img)
                    found_tile = True
                    break
            
            if not found_tile:
                break

        if len(tile_images) > 0:
            discarded_tiles = self.discard_tile_model.predict(tile_images)
            if save_img:
                player_name = ["player", "right", "opposite", "left"][player]
                for i, img in enumerate(original_tile_images):
                    tile_name = discarded_tiles[i]
                    file_name = f"data/test/{player_name}/{tile_name}_{100*(player+1)+i}.png"
                    cv2.imwrite(file_name, img)
            if riichi_index >= 0:
                discarded_tiles[riichi_index] += "r"
            if calling_tile:
                discarded_tiles[len(discarded_tiles)-1] += "c"
            return discarded_tiles
        else:
            return []

    def get_doras(self):
        doras = []
        doras += self.get_wall_doras(side=0)
        doras += self.get_wall_doras(side=1)
        doras += self.get_wall_doras(side=2)
        doras += self.get_wall_doras(side=3)
        return doras

    def get_wall_doras(self, side=0):
        icon_width = 21 if side % 2 == 0 else 29
        icon_height = 24 if side % 2 == 0 else 19
        tile_width = 25 if side % 2 == 0 else 33
        tile_height = 28 if side % 2 == 0 else 21
        x = [104, 498, 475, 73][side]
        y = [409, 389, 52, 77][side]

        doras = []

        while True:
            if sum(self.board[y, x]) < 140 or \
                sum(self.board[y+icon_height, x]) < 140 or \
                sum(self.board[y, x + icon_width]) < 140 or \
                sum(self.board[y+icon_height, x+icon_width]) < 140:
                # Found no tile
                match side:
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
                if side == 0:
                    x += 2
                elif side == 1:
                    y -= 1
                elif side == 2:
                    x -= 1
                tile_img = self.board[y:(y+icon_height), x:(x+icon_width)]
                match side:
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
                tile_img = self.discard_tile_model.preprocess(tile_img)

                doras.append(tile_img)

        if len(doras) == 0:
            return []

        return self.discard_tile_model.predict(doras)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Needs to receive the name of a file from the boards directory.")
        print("The filename should be without the directory and extension.")
        print("Usage: python tenhou/read_board.py <filename>")
        sys.exit(-1)
    main(sys.argv[1])