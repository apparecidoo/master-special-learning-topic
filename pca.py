import numpy as np
import matrix as _matrix
import mathematic as _math

def Pca(matrix):
    print(len(matrix))
    if(len(matrix) == 2):
        return PcaCustomFor2Variables(matrix)
    else:
        return PcaNumpy(matrix)

# função própria que calcula pca com duas variáveis
def PcaCustomFor2Variables(matrix):
    #subtract average from each item
    xAv = _math.AverageList(matrix[0])
    yAv = _math.AverageList(matrix[1])    
    for i in range(len(matrix[0])):
        matrix[0][i] = matrix[0][i] - xAv
    for i in range(len(matrix[1])):
        matrix[1][i] = matrix[1][i] - yAv

    matrix = _matrix.Transpose(matrix)
    covMatrix = _matrix.Covariance(matrix)
    eValues = EigenValues(covMatrix)
    eVector = EigenVector(covMatrix, eValues)

    return eValues, _matrix.Transpose(eVector)

# função que calcula o PCA utilizando função prontas do python
def PcaNumpy(matrix):
    covMatrix = _matrix.Covariance(_matrix.Transpose(matrix))
    eigenValues, eigenVectors = np.linalg.eig(covMatrix)
    auxZip = sorted(zip(eigenValues, eigenVectors.T))
    
    eVector = []
    eValues = []    
    for i in auxZip:
        eValues.append(i[0])
        eVector.append(i[1])
   
    eValues = eValues[::-1]
    eVector = np.asarray(eVector[::-1])
    
    return eValues, eVector

def EigenValues(matrix):

    if(len(matrix[0]) == 2):
        # Equation for two variables: X² - (d+a)X + ad = 0
        a = 1
        b = - (matrix[0][0] + matrix[1][1])
        c = (matrix[0][0] * matrix[1][1]) - (matrix[1][0] * matrix[0][1])

        return _math.Bhaskara(a, b, c)
    else:
        if(len(matrix[0]) == 3):
            a = 1
            b = matrix[2][2] + matrix[0][0] + matrix[1][1]
            c = matrix[1][2] * matrix[2][1] + matrix[0][1] * matrix[1][0] + matrix[2][0] * matrix[0][2] - ( matrix[0][0] * matrix[2][2] + matrix[1][1] * matrix[2][2] + matrix[0][0] * matrix[1][1])
            d = matrix[0][0] * matrix[1][1] * matrix[2][2] + matrix[0][1] * matrix[1][2] * matrix[2][0] + matrix[0][2] * matrix[1][0] * matrix[2][1] - ( matrix[2][0] * matrix[0][2] * matrix[1][1] + matrix[1][2] * matrix[2][1] * matrix[0][0] + matrix[0][1] * matrix[1][0] * matrix[2][2])
            return _math.CubicSolve(a, b, c, d)
        else:
            raise ArithmeticError("Cannot calculate Eigen Value different of 2x2 or 3x3 matrix.")

def EigenVector(cov, eValues):
    if(len(cov) == 2):
        eVec = []
        for i in eValues:
            res = _math.SolveLinearEquation(cov, i)
            eVec.append(res)
        return eVec
    else:
        raise ArithmeticError("Cannot calculate Eigen Value different of 2x2 matrix.")

def PcaTransformation(matrix, eigVec):
    return np.matmul(eigVec, matrix)
