from rref import *

def test():
    m1 = [[1,0,0],[0,1,0],[0,0,1]]
    reduced = rref(m1)

    m2 = [[3,12,15],[6,4,1],[1,1,5]]
    reduced = rref(m2)
    #print(reduced)

    m3 = [[1,1,1],[1,1,1]]
    reduced =rref(m3)
    #print(reduced)

    m4 = [[0,0,0],[1,1,1]]
    reduced = rref(m4)
    print(reduced)

test()
