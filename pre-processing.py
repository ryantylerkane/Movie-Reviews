#pre-process.py
#This program takes a training or test directory as input, performs pre-processing steps,
#and saves the contents of the files in a vector format to be used in NB.py

#The pre-processing steps will separate punctuation from words, make all characters
#and lowercase each character. Punctuation and words (if any) that do not appear in the vocabulary
#file will not be included in the output vector.

import sys #Needed to obtain training/test directory from the command line input parameters.
import os #Needed navigate to provided directory and read all of its files.
import re #Needed to identify punctuation during preprocessing.

def loadVocabulary(): #Function extracts each word from the vocabulary file and places them into the array. The indices of these values will be used to determine where the counts for each word are placed in the feature vectors.
    inputHandler = open("imdb.vocab", 'r', encoding='unicode_escape') #Open the vocab file located in the same directory as this program.
    vocabArray = {}
    for term in inputHandler:
        newTerm = term.replace("\n","")
        vocabArray[newTerm] = 0 #Append the term to the end of the array. This will allow the array to hold each word in the same order as the file.
    inputHandler.close()
    return vocabArray

def processVectorBOW(termString, vocabulary): #Function will check if a term belongs to the vocabulary and update the corresponding value in the document vector if necessary.
    if termString in vocabulary.keys(): #Update the count of that documents feature vector.
        vocabulary[termString] += 1 #Increase the count of that term in the document by one.
     #If the token is not in the vocabulary, it can be ignored.

def writeVector(vocabulary, classType, outputHandler): #Function records the class and feature vector that was obtained through preprocessing to an output file.
    outputHandler.write(classType + " ")
    for key in vocabulary.keys():
        if vocabulary[key] >0:
            outputHandler.write(key + "~" + str(vocabulary[key]) + " ") #Write each element of the feature vector with a space between each index. Use ~ to separate word from frequency as "=" is a character found in some words in vocabulary.
            vocabulary[key]=0
    outputHandler.write("\n") #Skip to the next line once the feature vector has been completed.
        
def processDirectory(vocabulary):    
    fileCount=0 #Integer that will allow us to count number of files processed (for debugging purposes).

    directory = sys.argv[1] #Extract the directory from the command line argument.

    print("Directory inputted: " + directory+'\n')

    folders = directory.split('\\') #Split each sub-directory into its own word so we can extract the class name and determine whether we received a test or training directory. 
    #classType = folders[-1] #Use if we are providing just a negative or just a positive training/test directory.
    classes = ["pos", "neg"] #Both test and training directories will have positive and negative sub-directories.
    folderType = folders[-1] #Test or training folder.

    outputFileName = folderType +".txt" #Create an output file where the name indicates whether it is a training or test directory.
    outputHandler = open(outputFileName, 'w') #Open output file for writing.
    for posOrNeg in classes:
        for file in os.listdir(sys.argv[1]+"\\"+posOrNeg): #For each file in the test or training directory. 
            fileCount +=1 #Increment the count for debugging purposes.
            try:
                inputHandler = open(directory+"\\"+ posOrNeg+"\\"+file,'r', encoding='unicode_escape') #Open input file for reading. Need to use unicode character encoding to avoid errors used with default encoding.
                docVector = [0] * len(vocabulary) #Create a new features vector for the document that was just opened.
                for line in inputHandler: #For each line in the training/test file.
                    lowerLine = line.lower() #Convert uppercase letters to lowercase.
                    tokens = lowerLine.split() #Split on whitespace.
                    for token in tokens:
                        #Attempt to capture series of symbols such as emoticons that were deemed significant enough to appear in the vocabulary. These are primarily continuous sequences of symbols, but there are a few that contained letters and numbers which are encompossed in a separate list. 
                        if re.match(r"^\W+$", token) or token in {"8)", ":o)", "=o)","8(","=p","=d",":d",":p",":-d",":-p"}: #If the token is strictly symbols, leave it as is. This will allow us to match emoticons found in the vocabulary without splitting the symbols. Note that this leaves us the possibility of not breaking up strings like "..." and "--" between words. These are not part of the vocabulary and will be discarded anyway.
                            processVectorBOW(token, vocabulary)
                        else: #Token is either a word containg only letters or a mixture of words and letters.
                                #Note that the vocabulary file contains words with "'" and "'" in the middle of their strings. Normally we would want to split words on these characters even if they are in the middle of the string.
                                #However, since it was not done in the test file, it is best to leave them as is for easier matching to the vocabulary array.
                            processedTokens=re.findall(r"[\w'-]+|[.,!?;:\"/&~<>()_]", token)

                                #Using this regex, strings where "-" or "'" are in the first or last position won't be split. We must check for these prior to writing to the features vector.
                            for pToken in processedTokens:
                                newTokens=[]
                                if len(pToken) > 1 and (pToken[0] == "'" or pToken[0]=="-"):
                                    newTokens.append(pToken[:1])
                                    pToken = pToken[1:]
                                if len(pToken) > 1 and (pToken[-1] == "'" or pToken[-1]=="-"): #Can't make this elif, as a string by have "'" in the beginning and the end.
                                    newTokens.append(pToken[:(len(pToken)-1)])
                                    pToken=pToken[(len(pToken)-1):]

                                if len(newTokens) ==0: #No splitting was required on the string, it can be processed as is.
                                    processVectorBOW(pToken, vocabulary)
                                else: #Strings were split, we have to process the new pieces.
                                    newTokens.append(pToken)
                                    for t in newTokens:
                                        processVectorBOW(t, vocabulary)
                writeVector(vocabulary, posOrNeg, outputHandler)
                inputHandler.close()
                
            except Exception as ex:
                raise ex
                print("Files unable to be read in directory: " + directory)
    outputHandler.close()

vocabulary = loadVocabulary()   
processDirectory(vocabulary)
