WINDS = ["東", "南", "西", "北"]

TILES = """
    1m 2m 3m 4m 5m 6m 7m 8m 9m
    1p 2p 3p 4p 5p 6p 7p 8p 9p
    1s 2s 3s 4s 5s 6s 7s 8s 9s
    ew sw ww nw
    wd gd rd
""".split()

CHII_COMBINATIONS = {
    "1": ["23"],
    "2": ["13", "34"],
    "3": ["12", "24", "45"],
    "4": ["23", "35", "56"],
    "5": ["34", "46", "67"],
    "6": ["45", "57", "78"],
    "7": ["56", "68", "89"],
    "8": ["67", "79"],
    "9": ["78"],
}