def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    """Make divisible function.

    This function rounds the channel number down to the nearest value that can
    be divisible by the divisor.

    Args:
        value (int): The original channel number.
        divisor (int): The divisor to fully divide the channel number.
        min_value (int, optional): The minimum value of the output channel.
            Default: None, means that the minimum value equal to the divisor.
        min_ratio (float): The minimum ratio of the rounded channel
            number to the original channel number. Default: 0.9.
    Returns:
        int: The modified output channel number
    """

    if min_value is None:
        min_value = divisor#8
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)#18+8/2=22,22//8*8=16
    # Make sure that round down does not go down by more than (1-min_ratio).
    if new_value < min_ratio * value:#0.9*18
        new_value += divisor#16+8
    return new_value

if __name__ == "__main__":
    print(make_divisible(72//4,8))#18,8