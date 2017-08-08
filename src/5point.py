from svd import *

original_points = [[ 0.4964,  1.0577],
                   [ 0.3650, -0.0919],
                   [-0.5412,  0.0159],
                   [-0.5239,  0.9467],
                   [ 0.3467,  0.5301],
                   [ 0.2797,  0.0012],
                   [-0.1986,  0.0460],
                   [-0.1622,  0.5347],
                   [ 0.0796,  0.2379],
                   [-0.3946,  0.7969]]

final_points = [[ 0.7570, 2.7340],
                [ 0.3961, 0.6981],
                [-0.6014, 0.7110],
                [-0.7385, 2.2712],
                [ 0.4177, 1.2132],
                [ 0.3052, 0.4835],
                [-0.2171, 0.5057],
                [-0.2059, 1.1583],
                [ 0.0946, 0.7013],
                [-0.6236, 3.0253]]

def solve5PointEssential(original_points, final_points, num_points):
    if num_points < 5:
        return False

    M = [[0] * 9 for i in range(num_points)]
    for i in range(num_points):
        x1 = original_points[i][0]
        y1 = original_points[i][1]

        x2 = final_points[i][0]
        y2 = final_points[i][1]

        M[i][0] = x1*x2
        M[i][1] = x2*y1
        M[i][2] = x2
        M[i][3] = x1*y2
        M[i][4] = y1*y2
        M[i][5] = y2
        M[i][6] = x1
        M[i][7] = y1
        M[i][8] = 1.0

    