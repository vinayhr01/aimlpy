import csv
import math

def loadcsv(filename):
    dataset = list(csv.reader(open(filename,"r")))
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet,testSet = dataset[:trainSize],dataset[trainSize:]
    return [trainSet, testSet]

def mean(numbers):
    return sum(numbers)/(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    v = 0
    for x in numbers:
        v += (x-avg)**2
    return math.sqrt(v/(len(numbers)-1))

def summarizeByClass(trainingSet):
    separated = {}
    for i in range(len(trainingSet)):
        vector = trainingSet[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = [(mean(attribute), stdev(attribute)) for attribute in zip(*instances)][:-1]
    return summaries

def calculateProbability(x, mean, stdev):
    exponent = math.exp((-(x-mean)**2)/(2*(stdev**2)))
    return (1 / math.sqrt(2*math.pi*(stdev**2))) * exponent

def predict(summaries, testSetInstance):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            probabilities[classValue] *= calculateProbability(testSetInstance[i], mean, stdev)
            bestLabel, bestProb = None, -1
            for classValue, probability in probabilities.items():
                if bestLabel is None or probability > bestProb:
                    bestProb = probability
                    bestLabel = classValue
    return bestLabel

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct/(len(testSet))) * 100.0

filename = 'tsvnaive.csv'
splitRatio = 0.9
dataset = loadcsv(filename)
actual = []
trainingSet, testSet = splitDataset(dataset, splitRatio)
for i in range(len(testSet)):
	actual.append(testSet[i][-1])
print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
summaries = summarizeByClass(trainingSet) #will have (mean,sd) for all attributes.(for class 1 & 0 separately)
predictions = getPredictions(summaries, testSet)
print('\nActual values:\n',actual)
print("\nPredictions:\n",predictions)
accuracy = getAccuracy(testSet, predictions)
print("Accuracy",accuracy)