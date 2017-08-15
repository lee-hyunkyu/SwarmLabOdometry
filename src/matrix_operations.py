def transpose(m):
    t = []
    for r in range(len(m)):
        tRow = []
        for c in range(len(m[r])):
            if c == r:
                tRow.append(m[r][c])
            else:
                tRow.append(m[c][r])
        t.append(tRow)
    return t

def getMatrixMinor(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

def determinant(m):
    #base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]

    d = 0
    for c in range(len(m)):
        d += ((-1)**c)*m[0][c]*determinant(getMatrixMinor(m,0,c))
    return d

def inverse(m):
    d = determinant(m)
    #special case for 2x2 matrix:
    if len(m) == 2:
        return [[m[1][1]/d, -1*m[0][1]/d],
                [-1*m[1][0]/d, m[0][0]/d]]

    #find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m,r,c)
            cofactorRow.append(((-1)**(r+c)) * determinant(minor))
        cofactors.append(cofactorRow)
    cofactors = transpose(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/determinant
    return cofactors

def matmul(A, B):
    if len(A[0]) != len(B):
        raise TypeError("Invalid matrix dimensions")
    return [[dot(row, col) for col in transpose(B)] for row in A]

def dot(a, b):
    if isinstance(a, int):
        return [a * x for x in b]
    elif isinstance(b, int):
        return [b * x for x in a]  
    elif len(a) != len(b):
        raise TypeError("Invalid vector dimensions")
    return sum(a[i] * b[i] for i in range(len(a)))

def norm(M):
    if isinstance(M[0], list):
        x = 0
        for row in range(len(M)):
            for col in range(len(M[row])):
                x += M[row][col]*M[row][col]
        return sqrt(x)
    else:
        return sqrt(dot(M, M))

def outer(a, b):
    return [[i * j for j in b] for i in a]