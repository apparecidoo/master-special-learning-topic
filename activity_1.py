import numpy
import matrix as _matrix
import least_squares as leastSquares

class Activity1:
    def __init__(self):
        pass

    def dataset_test(self, path, prediction_array = None, datatype="float"):
        prediction_test = []
        if(prediction_array != None):
            prediction_test = _matrix.ConvertArrayToMatrix(prediction_array)
        
        print(">>>>>> " + path + " <<<<<<")
        readmatrix = _matrix.ReadCsv(path, ";", datatype)
        matrix = numpy.delete(readmatrix, len(readmatrix[0]) - 1, 1) #extract features    
        # extract y
        y = _matrix.Copy(readmatrix)
        for col in range(len(readmatrix[0]) - 1):
            y = numpy.delete(y, 0, 1)        
        
        result_lin = leastSquares.LinearLeastSquares(matrix, y)
        _matrix.matrix_print("\n Beta for LinearLeastSquares", result_lin)
        if(prediction_array != None):
            _matrix.matrix_print(">> Prediction for: ", prediction_test)
            print("Is: " + str(leastSquares.LinearPredict(prediction_test, result_lin)))
            
        result_quad = leastSquares.QuadraticLeastSquares(matrix, y)
        _matrix.matrix_print("\n Beta for QuadraticLeastSquares", result_quad)
        if(prediction_array != None):
            _matrix.matrix_print(">> Prediction for: ", prediction_test)
            print("Is: " + str(leastSquares.QuadraticPredict(prediction_test, result_quad)))
            
        result_robust = leastSquares.RobustLeastSquares(matrix, y)
        _matrix.matrix_print("\n Beta for RobustLeastSquares", result_robust)
        if(prediction_array != None):
            _matrix.matrix_print(">> Prediction for: ", prediction_test)
            print("Is: " + str(leastSquares.LinearPredict(prediction_test, result_robust)))
        
        print("\n--------------------------------------------------")
    
    def test(self):
        self.dataset_test("datasets/alpswater.csv", [190])
        self.dataset_test("datasets/books_attend_grade.csv", [0, 9])
        self.dataset_test("datasets/us_census.csv", [2010.0])