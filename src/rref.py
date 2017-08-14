"""
Performs Gauss-Jordan elimination
"""

EPSILON = 0.005

def rref(lst):
    """
    Takes a 2D list and row reduces it using partial pivoting
    """
    reduced = lst[:]
    for i in range(len(reduced)):
        reduced[i] = invertRow(reduced[i])
        row = reduced[i]
        pivotIndex = getPivotIndex(row)
        if pivotIndex == -1:
            continue

        for j in range(len(reduced)):
            if (i == j or reduced[j][pivotIndex] == 0):
                continue

            scaleFactor = reduced[j][pivotIndex] * -1.0
            newRow = scaleRow(reduced[i], scaleFactor)
            newRow = addRows(newRow, reduced[j])
            reduced[j] = newRow

    swapRows(reduced)
    clean(reduced)
    return reduced



"""
***** Begin helper functions for rref *****
"""
def shape(lst):
    return (len(lst), len(lst[0]))


def swap(lst, i, j):
    temp = []
    for elem in lst[i]:
        temp.append(elem)
    lst[i] = lst[j]
    lst[j] = temp

def swapRows(lst):
    for i in range(len(lst)):
        pivotIndex = getPivotIndex(lst[i])

        for j in range(i, len(lst)):
            if i == j:
                continue
            
            otherPivotIndex = getPivotIndex(lst[j])
            if otherPivotIndex == -1:
                continue

            if (pivotIndex > otherPivotIndex) or (pivotIndex == -1):
                swap(lst, i, j)
                i = 0

def invertRow(row):
    for i in range(len(row)):
        if abs(row[i]) > EPSILON:
            factor = 1/row[i]
            row = scaleRow(row, factor)
    return row

def scaleRow(row, factor):
    newRow = []
    for elem in row:
        toAppend = elem * factor
        newRow.append(toAppend)
    return newRow

def addRows(row1, row2):
    newRow = []
    for i in range(len(row1)):
        newRow.append(row1[i] + row2[i])
    return newRow

def getPivotIndex(row):
    for i in range(len(row)):
        if abs(row[i] - 1) < EPSILON:
            return i
    return -1 # No pivots

def clean(lst):
    rows, cols = shape(lst)
    for i in range(rows):
        for j in range(cols):
            if abs(lst[i][j]) < EPSILON:
                lst[i][j] = 0
            else:
                lst[i][j] = int(lst[i][j] * 1000) / 1000
