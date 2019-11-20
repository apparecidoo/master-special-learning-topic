import csv
import numpy as np
import mathematic as _math

def ReadCsv(path, delimiter=";", mtype="float"):
    reader = csv.reader(open(path, "r"), delimiter = delimiter)
    x = list(reader)
    return np.array(x).astype(mtype)

def matrix_print(Title, M, roundVal = 3):
    print(Title)
    for row in M:
        print([round(x,roundVal)+0 for x in row])

def PrintTwo(Action, Title1, M1, Title2, M2):
    print(Action)
    print(Title1, '\t'*int(len(M1)/2)+"\t"*len(M1), Title2)
    for i in range(len(M1)):
        row1 = ['{0:+7.3f}'.format(x) for x in M1[i]]
        row2 = ['{0:+7.3f}'.format(x) for x in M2[i]]
        print(row1,'\t', row2)

def Create(rows, cols, value = 0.0):
    A = []
    for i in range(rows):
        A.append([])
        for j in range(cols):
            A[-1].append(value)

    return A

def Copy(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    matrixcp = Create(rows, cols)

    for i in range(rows):
        for j in range(cols):
            matrixcp[i][j] = matrix[i][j]

    return matrixcp

def ConvertArrayToMatrix(array):
    result = Create(1, len(array))
    for col in range(len(array)):
        result[0][col] = array[col]

    return result

def MultiplicationEscalarMatrix(matrix_a, matrix_b):
    result = Copy(matrix_a)
    for row in range(len(matrix_a)):
        for col in range(len(matrix_a[0])):
            result[row][col] = matrix_a[row][col] * matrix_b[row][0]
    return result

def RemoveColumn(matrix, pos):
    rows = len(matrix)
    cols = len(matrix[0])
    newcols = len(matrix[0]) - 1
    newmatrix = Create(rows, newcols)

    for i in range(rows):
        for j in range(cols):
            if(j == pos):
                continue
            else:
                if(j > pos):
                    newmatrix[i][j - 1] = matrix[i][j]
                else:
                    newmatrix[i][j] = matrix[i][j]

    return newmatrix

def AddColumn(matrix, pos, value = 0):
    rows = len(matrix)
    cols = len(matrix[0])
    newcols = len(matrix[0]) + 1
    newmatrix = Create(rows, newcols)

    for i in range(rows):
        for j in range(cols):
            if(j == pos):
                newmatrix[i][j] = value
            else:
                if(j > pos):
                    newmatrix[i][j + 1] = matrix[i][j]
                else:
                    newmatrix[i][j] = matrix[i][j]

    return newmatrix

def AddBeginColumn(matrix, value):
    return AddColumn(matrix, 0, value)

def AddEndColumnPotentialLast(matrix, potential):
    rows = len(matrix)
    cols = len(matrix[0])
    newcols = len(matrix[0]) + 1
    newmatrix = Create(rows, newcols)

    for i in range(rows):
        newmatrix[i][-1] = matrix[i][-1]**potential

    for i in range(rows):
        for j in range(cols):
            newmatrix[i][j] = matrix[i][j]

    return newmatrix

def CheckSquareness(A):
    if len(A) != len(A[0]):
        raise ArithmeticError("Matrix must be square to inverse.")

def Determinant(A, total=0):
    indices = list(range(len(A)))

    if len(A) == 2 and len(A[0]) == 2:
        val = A[0][0] * A[1][1] - A[1][0] * A[0][1]
        return val

    for fc in indices:
        As = Copy(A)
        As = As[1:]
        height = len(As)

        for i in range(height):
            As[i] = As[i][0:fc] + As[i][fc+1:]

        sign = (-1) ** (fc % 2)
        sub_det = Determinant(As)
        total += A[0][fc] * sign * sub_det

    return total

def CheckNonSingular(A):
    det = Determinant(A)
    if det != 0:
        return det
    else:
        raise ArithmeticError("Singular Matrix!")

def CheckEquality(A,B, tol=None):
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        return False

    for i in range(len(A)):
        for j in range(len(A[0])):
            if tol == None:
                if A[i][j] != B[i][j]:
                    return False
            else:
                if round(A[i][j],tol) != round(B[i][j],tol):
                    return False

    return True

def Multiplication(matrix_a, matrix_b):
    rows_a = len(matrix_a)
    cols_a = len(matrix_a[0])
    rows_b = len(matrix_b)
    cols_b = len(matrix_b[0])

    if cols_a != rows_b:
        print("Incorrect dimensions.")
        return

    result = [[0 for row in range(cols_b)] for col in range(rows_a)] #creating result

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return result

def Subtraction(matrix_a, matrix_b):
    rows_a = len(matrix_a)
    cols_a = len(matrix_a[0])
    rows_b = len(matrix_b)

    if rows_a != rows_b:
        print("Incorrect dimensions.")
        return

    result = Create(rows_a, cols_a)

    for i in range(rows_a):
        for j in range(cols_a):
            result[i][j] += matrix_a[i][j] - matrix_b[i][j]
    return result

def Sum(matrix_a, matrix_b):
    rows_a = len(matrix_a)
    cols_a = len(matrix_a[0])
    rows_b = len(matrix_b)

    if rows_a != rows_b:
        print("Incorrect dimensions.")
        return

    result = Create(rows_a, cols_a)

    for i in range(rows_a):
        for j in range(cols_a):
            result[i][j] += matrix_a[i][j] + matrix_b[i][j]
    return result

def Inverse(matrix_a, tol=None):
    CheckSquareness(matrix_a)
    CheckNonSingular(matrix_a)
    n = len(matrix_a)
    AM = Copy(matrix_a)
    I = Identity(n)
    IM = Copy(I)
    indices = list(range(n))
    for fd in range(n):
        fdScaler = 1.0 / AM[fd][fd]
        for j in range(n):
            AM[fd][j] *= fdScaler
            IM[fd][j] *= fdScaler
        for i in indices[0:fd] + indices[fd+1:]: 
            crScaler = AM[i][fd]
            for j in range(n): 
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
                IM[i][j] = IM[i][j] - crScaler * IM[fd][j]

    if CheckEquality(I, Multiplication(matrix_a,IM), tol):
        return IM
    else:
        raise ArithmeticError("Matrix inverse out of tolerance.")

def Identity(n):
    matrix_iden = Create(n, n)
    for i in range(n):
        matrix_iden[i][i] = 1
    return matrix_iden

def Transpose(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    matrixtrans = Create(cols, rows)

    for i in range(rows):
        for j in range(cols):
            matrixtrans[j][i] = matrix[i][j]

    return matrixtrans

def Covariance(matrix):
    if(len(matrix[0]) == 2):
        x = Transpose(matrix)[0]
        y = Transpose(matrix)[1]
        return [[_math.Covariance(x,x), _math.Covariance(x,y)], [_math.Covariance(y,x), _math.Covariance(y,y)]]
    else:
        if(len(matrix[0]) == 3):
            x = Transpose(matrix)[0]
            y = Transpose(matrix)[1]
            z = Transpose(matrix)[2]
            return [[_math.Covariance(x,x), _math.Covariance(x,y), _math.Covariance(x,z)], [_math.Covariance(y,x), _math.Covariance(y,y), _math.Covariance(y,z)], [_math.Covariance(z,x), _math.Covariance(z,y), _math.Covariance(z,z)]]
        else:
            raise ArithmeticError("Cannot calculate Eigen Value different of 2x2 or 3x3 matrix.")

def LinearSystem(matrix):
    A = RemoveColumn(matrix, len(matrix[0]) - 1)
    B = Transpose([Transpose(matrix)[-1]])
    AM = Copy(A)
    n = len(A)
    BM = Copy(B)
    
    indices = list(range(n))
    for fd in range(n):
        fdScaler = 1.0 / AM[fd][fd]
        for j in range(n):
            AM[fd][j] *= fdScaler
        BM[fd][0] *= fdScaler
        
        for i in indices[0:fd] + indices[fd+1:]:
            crScaler = AM[i][fd]
            for j in range(n):
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
            BM[i][0] = BM[i][0] - crScaler * BM[fd][0]

    return Multiplication(A, BM)