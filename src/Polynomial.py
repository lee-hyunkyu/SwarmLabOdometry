class Polynomial:
    def __init__(self, size = 10):
        self.coeffs = [0] * size

    def __len__(self):
        return len(self.coeffs)

    def __getitem__(self, degree):
        if degree < 0:
            raise IndexError("Index out of range")
        elif degree >= len(self):
            return 0
        return self[degree]

    def __setitem__(self, degree, value):
        if degree < 0:
            raise IndexError("Invalid index")
        elif degree >= len(self):
            self.coeffs += [0] * (degree + 1 - len(self))
        self[degree] = value

    def eval(self, x):
        output = 0
        temp_x = 1
        for i in range(len(self)):
            output += self[i] * temp_x
            temp_x *= x
        return output

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            output = Polynomial(len(self))
            for i in range(len(self)):
                output[i] = self[i] * other
            return output
        elif isinstance(other, Polynomial):
            output = Polynomial(len(self) * len(other))
            for i in range(len(self)):
                for j in range(len(other)):
                    c = self[i] * other[j]
                    output[i + j] += c
            return output
        raise TypeError("Multiplying Polynomial with invalid type")

    __rmul__ = __mul__

    def __add__(self, other):
        if isinstance(other, (int, float)):
            output = Polynomial(len(self))
            for i in range(len(self)):
                output[i] = self[i]
            output[0] += other
            return output
        elif isinstance(other, Polynomial):
            output = Polynomial(max(len(self), len(other)))
            for i in range(len(output)):
                output[i] = self[i] + other[i]
            return output
        raise TypeError("Adding polynomial with invalid type")

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            output = Polynomial(len(self))
            for i in range(len(self)):
                output[i] = self[i]
            output[0] -= other
            return output
        elif isinstance(other, Polynomial):
            output = Polynomial(max(len(self), len(other)))
            for i in range(len(output)):
                output[i] = self[i] - other[i]
            return output
        raise TypeError("Adding polynomial with invalid type")

class PolyMatrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = []
        for i in range(rows):
            self.data.append([])
            for j in range(cols):
                self.data[i].append(Polynomial())
    
    def __getitem__(self, pos):
        x, y = pos
        return self.data[x][y]

    def eval(self, x):
        result = [[0 for i in range(self.rows)] for j in range(self.cols)]
        for i in range(self.rows):
            for j in range(self.cols):
                result[i][j] = self[i, j].eval(x)
        return result