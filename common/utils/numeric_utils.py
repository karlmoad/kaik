from typing import Union


def parse_numeric(string:str) -> Union[int, float]:
    if string.find('.') != -1:
        return float(string)
    else:
        return int(string)