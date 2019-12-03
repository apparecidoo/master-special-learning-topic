import matrix as _matrix
import numpy as np
import random
import math 
import plot 

# calculate centroid
def GenerateCentroid(k, data, method = 'random'):
    if(method == 'random'):
        return RandomCentroid(k, data)
    else:
        return RandomCentroid(k, data)

def RandomCentroid(k, data):
    centroid = _matrix.Create(k, len(data[0]) + 1)
    dataGenerated = _matrix.Create(k, len(data[0]) + 1)

    for i in range(k):
        centroid[i][-1] = i
        randVal = random.choice(data)

        while(randVal[0] in np.transpose(dataGenerated)[0]): # get differents centroids
            randVal = random.choice(data)

        for j in range(len(randVal)):
            dataGenerated[i][j] = randVal[j]
            centroid[i][j] = randVal[j]
    
    return np.asarray(centroid)

# calculate distance
def DistancePoints(firstPoint, secondPoint, typeDist = 'euclidean'):
    if(typeDist == 'euclidean'):
        return EuclideanDistance(firstPoint, secondPoint)
    else:
        if(typeDist == 'manhattan'):
            return ManhattanDistance(firstPoint, secondPoint)
        else:
            return EuclideanDistance(firstPoint, secondPoint)

def EuclideanDistance(firstPoint, secondPoint):
    sub = np.subtract(secondPoint, firstPoint)
    sub = sub**2
    return math.sqrt(sum(sub))

def ManhattanDistance(firstPoint, secondPoint):
    sub = np.subtract(secondPoint, firstPoint)
    return sum(math.sqrt(sub**2))

# Stop Criteria
def StopCriteria(data, newData, criteria = 'default'):
    if(criteria == 'default'):
        return CheckEquals(data, newData)
    else:
        return CheckEquals(data, newData)

def CheckEquals(data, newData):
    return np.array_equal(np.transpose(data)[len(data[0]) - 1], np.transpose(newData)[len(newData[0]) - 1])

def UpdateCentroid(data, centroids):
    sumCentroid = np.asarray(_matrix.Create(len(centroids), len(centroids[0]) + 1))
    
    for i in range(len(centroids)):
        sumCentroid[i][len(sumCentroid[0]) - 2] = i

    for row in range(len(data)):
        for j in range(len(data[row]) - 1):
            sumCentroid[int(data[row][-1])][j] += data[row][j]
        sumCentroid[int(data[row][len(data[row]) - 1])][-1] += 1

    for row in range(len(sumCentroid)):
        for j in range(len(sumCentroid[row]) - 2):
            sumCentroid[row][j] = sumCentroid[row][j] / sumCentroid[row][-1] if sumCentroid[row][-1] >= 1 else sumCentroid[row][j]
    
    return np.asarray(_matrix.RemoveColumn(sumCentroid, -1))

def Classification(data, centroids):
    for row in range(len(data)): # for each row
        distance = 999999999999.0
        for i in range(len(centroids)): # for each centroid
            distAux = DistancePoints(np.asarray(_matrix.RemoveColumn(_matrix.Copy(data), -1))[row], np.asarray(_matrix.RemoveColumn(_matrix.Copy(centroids), -1)[i]))
            if(distAux < distance):
                distance = distAux
                data[row][-1] = centroids[i][-1]

def Kmeans(data, k, centroidMethod = 'random', stopCriteria = 'default', distanceMethod = 'euclidean'):
    _data = np.asarray(_matrix.AddColumn(_matrix.Copy(data), -1)) # create new column for classification
    _newData = np.asarray(_matrix.Copy(_data))
    centroids = GenerateCentroid(k, data, centroidMethod) # create matrix of centroids

    Classification(_newData, centroids) # classificating
    
    while(not StopCriteria(_data, _newData)):
        _data = np.asarray(_matrix.Copy(_newData))
        centroids = UpdateCentroid(_newData, centroids) # grouping by class
        Classification(_newData, centroids) # classificating

    return _newData, centroids
