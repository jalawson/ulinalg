# matrices and some linear algebra routines imlemented using
# regular Python list stuctures

# regular nested list version operations

#def print_matrix(x):
#    for i in x:
#        print(i)
def print_matrix(x):
    for i in range(len(x)):
        if i == 0:
            print('['+repr(x[i]))
        elif i == (len(x) - 1):
            print('',repr(x[i])+']')
        else:
            print('',x[i])

def is_square(x):
    # returns True if x is square
    sum([len(i)==len(x) for i in x])==len(x)

def zeros(m, n):
    return [[0 for j in range(n)] for i in range(m)]

def eye(m):
    return [[1 if i == j else 0 for j in range(m)] for i in range(m)]

def copy(x):
    return [[x[i][j] for j in range(len(x))] for i in range(len(x[0]))]

def transpose(x):
    return [[x[i][j] for i in range(len(x))] for j in range(len(x[0]))]

def dot(x,y):
    return sum([x[0][m]*y[0][m] for m in [i for i in range(len(x))]])

def matxmat(x,y):
    y = transpose(y)
    return [[ dot([i],[j]) for j in y] for i in x]

def matadd(x,y):
    return [[x[i][j] + y[i][j] for j in range(len(x[i]))] for i in range(len(x))]

def det2x2(x):
    #for j in range(len(x[0])-2,0,-1):
    res = []
    i = len(x) - 2
    for j in range(len(x)-1):
        #for k in [h for h in [j+1,j+2] if h < len(x)]:
        for k in range(j,len(x)-1):
            #print(x[i][j], x[i][k+1])
            #print(x[i+1][j], x[i+1][k+1])
            res.append((x[i][j] * x[i+1][k+1]) - (x[i][k+1] * x[i+1][j]))
    return res

def echelon(x):
    ''' Returns [det(x) and inv(x)]

        Operates on a copy of x
        Using elementary row operations convert X to an upper matrix
        the product of the diagonal = det(X)
        Continue to convert X to the identity matrix
        All the operation carried out on the original identity matrix
        makes it the inverse of X
    '''
    # divide each row element by [0] to give a one in the first position
    # (may have to find a row to switch with if first element is 0)
    x = copy(x)
    inverse = eye(len(x))
    sign = 1
    factors = []
    p = 0
    while p < len(x):
        #print_matrix(x)
        d = x[p][p]
        if abs(d) < 1e-30:
            # pivot == 0 need to swap a row
            # need to check if swap row also has a zero at the same position
            print('p=',p)
            np = 1
            while (p+np) < len(x) and x[p+np][p] < 1e-30:
                np = np + 1
                print('trying', np)
            if (p+np) == len(x):
                # singular
                print('giving up', p+np)
                return [0, []]
            z = x[p+np]
            x[p+np] = x[p]
            x[p] = z
            # do identity
            z = inverse[p+np]
            inverse[p+np] = inverse[p]
            inverse[p] = z
            # change sign of det
            sign = -sign
            continue
        factors.append(d)
        # change target row
        for n in range(p,len(x)):
            x[p][n] = x[p][n] / d
        # need to do the entire row for the inverse
        for n in range(len(x)):
            inverse[p][n] = inverse[p][n] / d
        # eliminate position in the following rows
        for i in range(p+1,len(x)):
            # multiplier is that column entry
            t = x[i][p]
            for j in range(p,len(x)):
                x[i][j] = x[i][j] - (t * x[p][j])
            for j in range(len(x)):
                inverse[i][j] = inverse[i][j] - (t * inverse[p][j])
        p = p + 1
    s = sign
    for i in factors:
        s = s * i # determinant
    # travel through the rows eliminating upper diagonal non-zero values
    for i in range(len(x)-1):
        # final row should already be all zeros except for the final position
        for p in range(i+1,len(x)):
            # multiplier is that column entry
            t = x[i][p]
            for j in range(i+1,len(x)):
                x[i][j] = x[i][j] - (t * x[p][j])
            for j in range(len(x)):
                inverse[i][j] = inverse[i][j] - (t * inverse[p][j])
    return [s, inverse]

def submatrix(x,i,j):
    # return a matrix made up from all elements except x[i][j]
    res = []
    for m in range(len(x)):
        if m != i:
            row = []
            for n in range(len(x)):
                if n != j:
                   row.append(x[m][n])
            res.append(row)
    return res 

def tests():

    x = [[1.2,2,3,5,0,3]]
    y = [[4,5,6,3,2,6.5]]

    x1 = [[1.2,2,3,5,0,3],[4,5,6,3,2,6.5]]
    y1 = [[4,5,6,3,2,6.5],[1.2,2,3,5,0,3]]

    x2 = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    y2 = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    x4 = [[1,2],[3,4]]
    x5 = [[1],[2]]

    x6 = [[3,0,2],[2,0,-2],[0,1,1]]
    # test x7
    # det = 24
    x7 = [[3,2,0,1],[4,0,1,2],[3,0,2,1],[9,2,3,1]]
    # test x8 zero in first position
    # det = 42
    x8 = [[0,2,0,1],[4,0,1,2],[3,0,2,1],[9,2,3,1]]
    # singular matrix det = 0
    x9 = [[1,2,3,4],[5,6,7,8],[-1,-2,-3,-4],[-5,-6,-7,-8]]

    print("dot product vectors", x , y)
    print(dot(x,y))

    print("matrix product x1 * y1.transpose")
    print_matrix(x1)
    print_matrix(transpose(y1))
    X = matxmat(x1,transpose(y1))
    print_matrix(X)

    print("matrix product x1.transpose() * y1")
    print_matrix(transpose(x1))
    print_matrix(y1)
    X = matxmat(transpose(x1),y1)
    print_matrix(X)

    print("matrix + matrix x2 + x2")
    print_matrix(x2)
    print_matrix(y2)
    X = matadd(x2, y2)
    print_matrix(X)

    print("det2x2 x4)", x4)
    print(det2x2(x4))
    print("det2x2 x7)", x7)
    print(det2x2(x7))

    x7 = [[3,2,0,1],[4,0,1,2],[3,0,2,1],[9,2,3,1]]
    print("submatrix x7,2,1")
    print_matrix(x7)
    print_matrix(submatrix(x7,2,1))

    print_matrix(zeros(4,5))

    print_matrix(eye(4))

    x6 = [[3,0,2],[2,0,-2],[0,1,1]]
    x7 = [[3,2,0,1],[4,0,1,2],[3,0,2,1],[9,2,3,1]]
    x7d,x7I = echelon(x7)
    print_matrix(x7I)
    print("Det(x7) =", x7d)
    x8 = [[0,2,0,1],[4,0,1,2],[3,0,2,1],[9,2,3,1]]
    x8d,x8I = echelon(x8)
    print_matrix(x8)
    print_matrix(x8I)
    print("Det(x8) =", x8d)
    # x9 singular
    x9 = [[1,2,3,4],[5,6,7,8],[-1,-2,-3,-4],[-5,-6,-7,-8]]
    x9d, x9I = echelon(x9)
    print_matrix(x9)
    print_matrix(x9I)
    print("Det(x9) =", x9d)
    # x10 singular
    x10 = [[1,2,3],[4,5,6],[5,7,9]]
    x10d, x10I = echelon(x10)
    print_matrix(x10)
    print_matrix(x10I)
    print("Det(x10) =", x10d)

def main():

    tests()

if __name__ == "__main__":
    main()
