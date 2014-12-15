import sys
import math

def produce_size_strings(begin, end, base=2, n_dims=3):
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

    if begin > end:
        print "unable to produce strings between %i and %i" % (begin, end)
        return value

    cardinals_exponents = range(begin, end)
    cardinals = [str(int(math.pow(base, c))) for c in cardinals_exponents]

    start_size = "x".join([cardinals[0]]*n_dims)
    value.append(start_size)

    end_size = "x".join([cardinals[-1]]*n_dims)
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
    
    sargv = sys.argv
    # TODO: maybe wanna give something in through sys.argv
    begin = 6
    end = 10
    base = 2
    if len(sargv) == 3:
        begin = int(sargv[-2])
        end = int(sargv[-1])

    if len(sargv) == 2:
        end = int(sargv[-1])

    if len(sargv) == 4:
        begin = int(sargv[-3])
        end = int(sargv[-2])
        base = int(sargv[-1])
        
    res = produce_size_strings(begin, end, base)
    if res:
        print "\n".join(res)
        sys.exit(0)
    else:
        sys.exit(1)
