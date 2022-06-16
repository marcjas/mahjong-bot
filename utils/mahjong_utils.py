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

def get_improvement_tile_lists(hand):
    shanten = Shanten()
    agari = Agari()
    improvement_tile_lists = []
    man, pin, sou, honors = convert_hand(hand)
    hand = TilesConverter.string_to_34_array(
        man=man, pin=pin, sou=sou, honors=honors
    )
    for i, tile_count in enumerate(hand):
        if tile_count > 0:
            hand[i] -= 1
            new_shanten = shanten.calculate_shanten(hand)
            improvement_tiles = [0] * 34
            if new_shanten == 0:
                for j in range(34):
                    hand[j] += 1
                    if agari.is_agari(hand):
                        improvement_tiles[i] = 4
                    hand[j] -= 1
            else:
                for j in range(34):
                    if hand[j] > 0:
                        hand[j] -= 1
                        for k in range(34):
                            if hand[k] < 4:
                                hand[k] += 1
                                potential_shanten = shanten.calculate_shanten(hand)
                                if potential_shanten < new_shanten:
                                    improvement_tiles[j] = 4
                                hand[k] -= 1
                        hand[j] += 1
            hand[i] += 1

            improvement_tile_lists.append(improvement_tiles)

    return improvement_tile_lists