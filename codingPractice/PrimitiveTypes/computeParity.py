### O(n)
def parity(x):
    result = 0
    bits = str(x)
    for _ in bits:
        if _ == '1':
            result = result ^ 1

    return result


assert parity(100) == 1
assert parity(1011) == 1
assert parity(111) == 1
assert parity(100101) == 1
assert parity(0) == 0


### O(number of 1s)
