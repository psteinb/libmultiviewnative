
def produce_size_strings(begin, end):

    """ function that produces 3D grid sizes in power-of-2 manner
    example: produce_size_strings(4, 6):
    16x16x16
    32x16x16
    32x32x16
    32x32x32
    64x32x32
    64x64x32
    64x64x64
    """

    value = []
    cardinals_exponents = range(begin, end)
    cardinals = [str(1 << c) for c in cardinals_exponents]

    start_size = "x".join([cardinals[0]]*3)
    print start_size

    end_size = "x".join([cardinals[-1]]*3)
    cardinal_idx = 1

    while (start_size != end_size) and (cardinal_idx < len(cardinals)):
        previous = start_size
        temp_li = start_size.split("x")

        for it in temp_li:

            if (it == cardinals[cardinal_idx-1]):
                temp_li[temp_li.index(it)] = cardinals[cardinal_idx]
                break

        start_size = "x".join(temp_li)

        if previous != start_size:
            value.append(start_size)
        else:
            cardinal_idx += 1

    return value

if __name__ == '__main__':
    # TODO: maybe wanna give something in through sys.argv
    res = produce_size_strings(6, 10)
    print "\n".join(res)
