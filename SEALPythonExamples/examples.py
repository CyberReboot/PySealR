import time
import random
import pickle
import threading
import numpy as np
import pandas as pd
import math
import seal
from seal import ChooserEvaluator, \
	Ciphertext, \
	Decryptor, \
	Encryptor, \
	EncryptionParameters, \
	Evaluator, \
	IntegerEncoder, \
	FractionalEncoder, \
	KeyGenerator, \
	MemoryPoolHandle, \
	Plaintext, \
	SEALContext, \
	EvaluationKeys, \
	GaloisKeys, \
	PolyCRTBuilder, \
	ChooserEncoder, \
	ChooserEvaluator, \
	ChooserPoly


def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]


def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated


def mean(numbers):
	if(numbers [0] == 0.0 or numbers [1] == 1.0):
		return 0
	
	value = 0
	plain = encoder.encode(value)
	add = Ciphertext()
	encryptor.encrypt(plain, add)
	ev_keys16 = EvaluationKeys()
	keygen.generate_evaluation_keys(16, ev_keys16)
	for x in numbers:
		evaluator.add(add, x)
		evaluator.relinearize(add, ev_keys16)

	size = float(len(numbers))
	size = (1 / size)
	div = encoder.encode (size)
	evaluator.multiply_plain (add, div)
	return add

def power (x, avg):
	evaluator.negate (avg)
	evaluator.add (x, avg)
	evaluator.multiply (x, x)
	return x


def stdev(numbers):
	if(numbers [0] == 0.0 or numbers [1] == 1.0):
		return 0

	avg = mean(numbers)
	value = 0
	plain = encoder.encode(value)
	add_var = Ciphertext()
	encryptor.encrypt(plain, add_var)
	ev_keys16 = EvaluationKeys()
	keygen.generate_evaluation_keys(16, ev_keys16)
	for x in numbers:
		res = power(x, avg)
		evaluator.add(add_var, res)
		evaluator.relinearize(add_var, ev_keys16)

	size = float(len(numbers) - 1)
	size = 1 / size
	div = encoder.encode (size)
	evaluator.multiply_plain (add_var, div)
	plain_result = Plaintext()
	decryptor.decrypt(add_var, plain_result)
	value1 = float(encoder.decode(plain_result))
	return math.sqrt(value1)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries
 
def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries
 
def calculateProbability(x, mean, stdev):
	dividend = power (x, mean)
	divisor = (2*math.pow(stdev,2))
	divisor = 1 / divisor
	divisor_encode = encoder.encode (divisor)
	evaluator.multiply_plain (dividend, divisor_encode)
	
	plain_div = Plaintext ()
	decryptor.decrypt (dividend, plain_div)
	value = float (encoder.decode (plain_div))

	exponent = math.exp(-(value))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
 
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
			
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
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
	return (correct/float(len(testSet))) * 100.0


#importing the dataset
print ("Importing dataset...")
data = pd.read_csv ('/app/Social_Network_Ads.csv')
dataset = data.iloc [:, [2, 3, 4]].values
#dataset1 = data.iloc [:, [2, 3]].values
print ("Done\n\n\n")

print ("Setting encryption parameters...")
parms = EncryptionParameters()
parms.set_poly_modulus("1x^16384 + 1")
parms.set_coeff_modulus(seal.coeff_modulus_128(16384))
parms.set_plain_modulus(1 << 12)
print ("Done\n\n\n")


context = SEALContext(parms)
#print_parameters(context);
encoder = IntegerEncoder(context.plain_modulus())
keygen = KeyGenerator(context)
public_key = keygen.public_key()
secret_key = keygen.secret_key()
encryptor = Encryptor(context, public_key)
evaluator = Evaluator(context)
decryptor = Decryptor(context, secret_key)
encoder = FractionalEncoder(context.plain_modulus(), context.poly_modulus(), 2048, 32, 3)

X, Y = splitDataset(dataset, 0.8)
size = len(X)
size_y = len(Y)

X1 = [[0 for x in range(2)] for y in range(size)]
Y1 = [[0 for x in range(2)] for y in range(size)]

'''
for i in range (size):
	X [i] [1] = X [i] [1] / 100.0

for i in range (size_y):
	Y [i] [1] = Y [i] [1] / 100.0

'''
#Feature Scaling
print ("Scaling features...")
from sklearn.preprocessing import StandardScaler
sc = StandardScaler ()
X1 = sc.fit_transform (X)
Y1 = sc.transform (Y)
print ("Done\n\n\n")

for i in range (size):
	for j in range (2):
		X [i] [j] = X1 [i] [j]

for i in range (size_y):
	for j in range (2):
		Y [i] [j] = Y1 [i] [j]


print ("Encrypting training set...")
encrypted_data = [[Ciphertext(parms) for x in range(3)] for y in range(size)]

for i in range (size):
    for j in range (2):
        encryptor.encrypt(encoder.encode(X [i] [j]), encrypted_data [i] [j])
    encrypted_data [i] [2] = X [i] [2]
print ("Done\n\n\n")

print ("Creating prediction model, (This will take a few minutes)...")
summaries = summarizeByClass(encrypted_data)
print ("Done\n\n\n")


print ("Encrypting training set...")
size = len(Y)
encrypted_data1 = [[Ciphertext(parms) for x in range(3)] for y in range(size)]

for i in range (size):
    for j in range (2):
        encryptor.encrypt(encoder.encode(Y [i] [j]), encrypted_data1 [i] [j])
    encrypted_data1 [i] [2] = Y [i] [2]
print ("Done\n\n\n")

print ("Making predictions, (This will take a few minutes)...")
predictions = getPredictions(summaries, encrypted_data1)
accuracy = getAccuracy(Y, predictions)
print ("Done\n\n\n")

print("Prediction accuracy: " + (str) (accuracy))

