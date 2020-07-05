#NB.py
#This program takes four input arguments in the following order:
#A training file, a test file, a file where the parameters of the resulting model
#will be saved, and the output file hwere predictions made by the classifier on the
#test data will be written.
#The last line in the output file will list the overall accuracy of the classifier
#on the test data.

#The program implements Naive Bayes utilizing the training and test data compiled by pre-processing.py. The
#classifier uses bag-of-words features with Add-one smoothing.

import sys #Needed to obtain input parameters from the command line.
import math #Needed to perform probability calculations in logspace.

#sys.argv[1] = training file
#sys.argv[2] = test file
#sys.argv[3] = parameters output file
#sys.argv[4] = predictions (output) file

def countVocabulary(): #Function extracts each word from the vocabulary file and places them into the array. The indices of these values will be used to determine where the counts for each word are placed in the feature vectors.
    inputHandler = open("imdb.vocab", 'r', encoding='unicode_escape') #Open the vocab file located in the same directory as this program.
    vocab = 0
    for term in inputHandler:
        vocab += 1
    inputHandler.close()
    return vocab

def collectCounts(classDicts, classes): #Function contructs construct feature probabilities and prior probabilities using the provided training file.
   
    inputHandler = open(sys.argv[1], 'r', encoding='unicode_escape') #Open the training file to read its contents.
    
    for line in inputHandler:
        counts = line.split() #Split the line on whitespace.
        if counts[0] not in classDicts.keys():
            classDicts[counts[0]] = {}
            classes[counts[0]] = 0
        currentClass = counts[0]
        classes[currentClass] += 1 #Increase the document count for this class.

        for c in counts[1:]:
            word, num = c.split('~') #Separate the word and its count from each other.
            if word not in classDicts[currentClass].keys():
                classDicts[currentClass][word] = int(num)
            else:
                classDicts[currentClass][word] += int(num) 
    inputHandler.close()

def writeTraining(outputHandler, classDicts, classDenominators):
    outputHandler.write("Probabilities of Context Features Include Add One Smoothing: \n")
    for key in classDicts.keys():
        for k in classDicts[key].keys():
            outputHandler.write("c(" + k + "|" + key+")=" + str(classDicts[key][k])+",")
            probWithAddOne = (classDicts[key][k]+1)/(classDenominators[key])
            classDicts[key][k] = probWithAddOne
            outputHandler.write("p(" + k + "|" + key+")=" + str(probWithAddOne)+"\n")
        outputHandler.write("\n"+"All values not seen in training will receive a probability of: " + str(1/classDenominators[key])+"\n\n")


def predictClass(classProbs, outputPredictions, actualClass, numPredictions, numRightLabels):
    outputPredictions.write("max(")
    for key in classProbs.keys():
        outputPredictions.write("p" + "(" + key + "| d" + str(numPredictions)+ ") = " + str(classProbs[key])+" ")
    outputPredictions.write(") = ")

    maxClass = max(classProbs, key = classProbs.get)
    outputPredictions.write(maxClass + " || Provided label: " + actualClass)
    
    if maxClass==actualClass:
        numRightLabels +=1 #Increase the number of correct guesses.
        
    return numRightLabels
    
def processTest(classDicts, classes, vocab):

    inputHandler = open(sys.argv[2], 'r', encoding='unicode_escape') #Open the testing file to read its contents.
    outputParameter = open(sys.argv[3], 'w') #Open the parameters output file.
    outputPredictions = open(sys.argv[4], 'w') #Open the predictions output file.
    outputPredictions.write("ALL PROVIDED VALUES ARE IN POSITIVE LOG SPACE: \n")
    numRightLabels = 0
    numPredictions = 0
    classDenominators = {} #Calculate the denominator for each class so it doesn't have to be done for every context feature calculation.

    #Calculate the prior probability values prior to performing calculations.
    numDocs = sum(classes.values())
    outputParameter.write("PRIOR PROBABILITIES: ")
    for key in classes.keys(): #For each class count.
        classes[key] = classes[key]/numDocs #Calculate the prior probability for a given class.
        classDenominators[key] = sum(classDicts[key].values()) + vocab #Calculate the denominators that will be needed for Add one smoothing.
        outputParameter.write("p(" + key+") = " + str(classes[key]) + " ")
    outputParameter.write("\n\n")
    writeTraining(outputParameter, classDicts, classDenominators) #Write model parameters to the file.
    
    for line in inputHandler:
        #We must perfrom the probability calculations for a given training document in log space to avoid underflow errors (which will occur with the provided test documents if done in linear space).
        #We would ideally use negative log as it allows us to represent the probabilities as positive values, but doing so would represent smaller probabilities as larger numbers, making it more
        #difficult to find the maximum value. As such, we will use positive log space.
        classProbs = {} #Will hold the total probabilities calculated for a given class.
        numPredictions +=1 #Increase the total number of files viewed.
        for key in classes.keys():
            classProbs[key] = math.log10(classes[key]) #Set the probability equal to the prior probability in order to begin building the product.
            
        counts = line.split() #Split the line on whitespace.
        actualClass = counts[0] #Obtain the class that the string that the document is associated with.
        
        for c in counts[1:]:
            word, num = c.split('~') #Separate the word and its count from each other.
            #We do not have to worry about a word not being found in the vocabulary, as all non-vocabulary words were removed during pre-processing.
            for key in classes.keys():
                if word not in classDicts[key].keys(): #If the word was not seen in the classes training.
                    probWithAddOne = 1/(classDenominators[key]) #Make the zero count equal to one due to Add-one smoothing.
                    
                else:
                    probWithAddOne = classDicts[key][word] #One is already added to the probabilities in all training words (see writeTraining function). 

                totalProb = math.log10(probWithAddOne) * int(num)
                classProbs[key] += totalProb #Keep running total of probability for that class.
                
        numRightLabels = predictClass(classProbs, outputPredictions, actualClass, numPredictions, numRightLabels)
        outputPredictions.write("\n")

    outputPredictions.write("Number of predictions made: " + str(numPredictions) + " Percent Accuracy: " + str(numRightLabels/numPredictions * 100) + "%")            
    outputParameter.close()
    outputPredictions.close()
    inputHandler.close()
    
classDicts = {} #Will allow us to access a class's word count by providing the class's name.
classes = {} #Will allow us to check the document count of a class by referencing its name.
vocab = countVocabulary()
collectCounts(classDicts, classes)
processTest(classDicts, classes, vocab)
