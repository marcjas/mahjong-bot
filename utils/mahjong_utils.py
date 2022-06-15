from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter
from mahjong.agari import Agari

TILES = """
    1m 2m 3m 4m 5m 6m 7m 8m 9m
    1p 2p 3p 4p 5p 6p 7p 8p 9p
    1s 2s 3s 4s 5s 6s 7s 8s 9s
    ew sw ww nw
    wd gd rd
""".split()

HONOR_TILES = "ew sw ww nw wd gd rd".split()

def convert_hand(hand):
    man = []
    pin = []
    sou = []
    honors = []
    for tile in hand:
        suit = tile[1:2]
        if suit == "m":
            man.append(tile[:1])
        elif suit == "p":
            pin.append(tile[:1])
        elif suit == "s":
            sou.append(tile[:1])
        elif suit == "z":
            honors.append(HONOR_TILES.index(tile))

    return ''.join(man), ''.join(pin), ''.join(sou), ''.join(honors)

def get_shanten(hand):
    shanten = Shanten()
    man, pin, sou, honors = convert_hand(hand)
    tiles = TilesConverter.string_to_34_array(
        man=man, pin=pin, sou=sou, honors=honors
    )
    return shanten.calculate_shanten(tiles)

def get_potential_shanten(hand):
    potential_shantens = []
    for i in range(len(hand)):
        new_hand = [hand[j] for j in range(len(hand)) if j != i]
        potential_shantens.append(get_shanten(new_hand))
    return potential_shantens

def is_agari(hand):
    agari = Agari()
    man, pin, sou, honors = convert_hand(hand)
    tiles = TilesConverter.string_to_34_array(
        man=man, pin=pin, sou=sou, honors=honors
    )
    return agari.is_agari(tiles)

def get_improvement_tiles(hand):
    improvement_tile_count = []
    for i in range(len(hand)):
        new_hand = [hand[j] for j in range(len(hand)) if j != i]
        new_shanten = get_shanten(new_hand)
        improvement_tiles = []
        if new_shanten == 0:
            for new_tile in TILES:
                maybe_winning_hand = new_hand + [new_tile]
                if is_agari(maybe_winning_hand):
                    improvement_tiles.append(new_tile)
        else:
            for j, tile in enumerate(new_hand):
                for new_tile in TILES:
                    if new_tile != tile:
                        new_hand[j] = new_tile
                        potential_shanten = get_shanten(new_hand)
                        if potential_shanten < new_shanten:
                            if new_tile not in improvement_tiles:
                                improvement_tiles.append(new_tile)

                new_hand[j] = tile

        count = 0
        for tile in improvement_tiles:
            count += 4 - len([j for j in hand if j == tile])
        improvement_tile_count.append(count)

    return improvement_tile_count