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
    centroid = _matrix.Create(k, 3)
    dataGenerated = _matrix.Create(k, 3)

    for i in range(k):
        centroid[i][2] = i
        randVal = random.choice(data)

        while(randVal[0] in np.transpose(dataGenerated)[0] and randVal[1] in np.transpose(dataGenerated)[1]): # get differents centroids
            randVal = random.choice(data)

        dataGenerated[i][0] = randVal[0]
        dataGenerated[i][1] = randVal[1]
        centroid[i][0] = randVal[0]
        centroid[i][1] = randVal[1]
    
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
    return math.sqrt((secondPoint[0] - firstPoint[0])**2 + (secondPoint[1] - firstPoint[1])**2)

def ManhattanDistance(firstPoint, secondPoint):
    return abs((secondPoint[0] - firstPoint[0]) + (secondPoint[1] - firstPoint[1]))

# Stop Criteria
def StopCriteria(data, newData, criteria = 'default'):
    if(criteria == 'default'):
        return CheckEquals(data, newData)
    else:
        return CheckEquals(data, newData)

def CheckEquals(data, newData):
    return np.array_equal(np.transpose(data)[len(data[0]) - 1], np.transpose(newData)[len(newData[0]) - 1])

def UpdateCentroid(data, centroids):
    sumCentroid = np.asarray(_matrix.Create(len(centroids), 4))
    
    for i in range(len(centroids)):
        sumCentroid[i][2] = i

    for row in range(len(data)):
        sumCentroid[int(data[row][2])][0] += data[row][0]
        sumCentroid[int(data[row][2])][1] += data[row][1]

        sumCentroid[int(data[row][2])][3] += 1

    for row in range(len(sumCentroid)):
        sumCentroid[row][0] = sumCentroid[row][0] / sumCentroid[row][3] if sumCentroid[row][3] >= 1 else sumCentroid[row][0]
        sumCentroid[row][1] = sumCentroid[row][1] / sumCentroid[row][3] if sumCentroid[row][3] >= 1 else sumCentroid[row][1]
    
    return np.asarray(_matrix.RemoveColumn(sumCentroid, len(sumCentroid[0])))

def Classification(data, centroids):
    for row in range(len(data)): # for each row
        distance = 999999999999.0
        for i in range(len(centroids)): # for each centroid
            distAux = DistancePoints([data[row][0], data[row][1]], [centroids[i][0], centroids[i][1]])
            if(distAux < distance):
                distance = distAux
                data[row][2] = centroids[i][2]

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
