This project implements a Naive Bayes classifier that is trained using 25,000 positive and negative reviews gathered from imdb.com. 
The classifier is then tested on an additional 25,000 test files.

pre-processing.py:
1. The vocabulary file must be named "imdb.vocab" and must appear in the same directory as the program file.
2. The program requires that the input directory contains "pos" and "neg" sub-folders, each containing their respctive test/train documents.
3. The output file will take the name of the last directory in the provided input path. This should be "test" or "train" if the same directory structure for the raw data is used.
4. Unicode encoding was required to be used for the input files due to the usage of some Unicode characters in the test/training reviews.

NB.py:
1. The prior probabilities and context features will be outputted in linear space, as there was no problem with underflow when ran on the testing/training corpus.
2. The probabilities found in the prediction files will be outputted in log space. Negative log space was avoided so that the "max" function could easily be used to determine the predicted class.
3. 3. Unicode encoding was required to be used for the input files due to the usage of some Unicode characters in the test/training reviews.