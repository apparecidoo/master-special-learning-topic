import numpy
import matrix as _matrix

def LinearLeastSquares(matrix, y): # B = (X^T * X)^-1 * X^t * y
    matrix = _matrix.AddBeginColumn(matrix, 1)
    matrixtranspose = _matrix.Transpose(matrix)
    section_1 = _matrix.Inverse(_matrix.Multiplication(matrixtranspose, matrix), 1) # (X^T * X)^-1
    section_2 = _matrix.Multiplication(section_1, matrixtranspose) # (X^T * X)^-1 * X^t
    section_3 = _matrix.Multiplication(section_2, y) # (X^T * X)^-1 * X^t * y
    
    return section_3

def QuadraticLeastSquares(matrix, y): # B = (X^T * X)^-1 * X^t * y
    matrix = _matrix.AddEndColumnPotentialLast(_matrix.AddBeginColumn(matrix, 1), 2)
    matrixtranspose = _matrix.Transpose(matrix)
    section_1 = _matrix.Inverse(_matrix.Multiplication(matrixtranspose, matrix), 1) # (X^T * X)^-1
    section_2 = _matrix.Multiplication(section_1, matrixtranspose) # (X^T * X)^-1 * X^t
    section_3 = _matrix.Multiplication(section_2, y) # (X^T * X)^-1 * X^t * y

    return section_3

def RobustLeastSquares(matrix, y): # B = (X^T * W.X)^-1 * X^t * W.y
    matrixlinear = LinearLeastSquares(matrix, y)
    newy = LinearPredictMatrix(matrix, matrixlinear)    
    w = _matrix.Copy(newy)
    
    # calculating w
    for i in range(len(w)):
        w[i][0] = 1/(abs(y[i][0] - newy[i][0]))
    
    y = _matrix.MultiplicationEscalarMatrix(y, w)
    matrix = _matrix.AddBeginColumn(matrix, 1)

    matrix = _matrix.MultiplicationEscalarMatrix(matrix, w)
    matrixtranspose = _matrix.Transpose(matrix)
    section_1 = _matrix.Inverse(_matrix.Multiplication(matrixtranspose, matrix), 1) # (X^T * X)^-1
    section_2 = _matrix.Multiplication(section_1, matrixtranspose) # (X^T * X)^-1 * X^t
    section_3 = _matrix.Multiplication(section_2, y) # (X^T * X)^-1 * X^t * y
    
    return section_3

def LinearPredict(matrix_prediction_test, b):     
    matrix = _matrix.AddBeginColumn(matrix_prediction_test, 1)
    result = _matrix.Multiplication(matrix, b)
    return result[0][0]

def QuadraticPredict(matrix_prediction_test, b):
    matrix = _matrix.AddEndColumnPotentialLast(_matrix.AddBeginColumn(matrix_prediction_test, 1), 2)
    result = _matrix.Multiplication(matrix, b)
    return result[0][0]

def LinearPredictMatrix(matrix, b):
    result = _matrix.Create(len(matrix), 1)
    for row in range(len(matrix)):
        result[row][0] = LinearPredict([matrix[row]], b)
    return result