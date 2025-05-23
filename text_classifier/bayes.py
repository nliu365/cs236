'''
To Use it, 
>>> import bayes
>>> listOPosts, listClasses = bayes.loadDataSet()
>>> myVocabList = bayes.createVocalList(listOPosts)
>>> trainMat=[]
>>> for postinDoc in listOPosts:
...     trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))
...
>>> p0V,p1V,pAb=bayes.trainNB0(trainMat,listClasses)
'''

from numpy import *

def loadDataSet():
  postingList = [
    ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
  ]
  classVec = [0, 1, 0, 1, 0, 1] # 1 is abusive, 0 not
  return postingList, classVec


def createVocabList(dataSet):
  vocalSet = set([])
  for document in dataSet:
    vocalSet = vocalSet | set(document)
  return list(vocalSet)


def setOfWords2Vec(vocalList, inputSet):
  returnVec = [0] * len(vocalList)
  for word in inputSet:
    if word in vocalList:
      returnVec[vocalList.index(word)] = 1
    else:
      print(f'the word {word} is not in my Vocabulary!')
  return returnVec


def trainNB0(trainMatrix, trainCategory):
  numTrainDocs = len(trainMatrix)
  numWords = len(trainMatrix[0])
  pAbusive = sum(trainCategory) / float(numTrainDocs)
  p0Num = zeros(numWords)
  print(f'p0Num: {p0Num}')
  p1Num = zeros(numWords)
  print(f'p1Num: {p1Num}')
  p0Denom = 0.0
  p1Denom = 0.0
  for i in range(numTrainDocs):
    if trainCategory[i] == 1:
      p1Num += trainMatrix[i]
      print(f'p1Num in for loop: {p1Num}')
      p1Denom += sum(trainMatrix[i])
      print(f'p1Denom in for loop: {p1Denom}')
    else:
      p0Num += trainMatrix[i]
      p0Denom += sum(trainMatrix[i])
  p1Vect = p1Num / p1Denom
  p0Vect = p0Num / p0Denom
  return p0Vect, p1Vect, pAbusive


def classifyNB(vex2Classify, p0Vec, p1Vec, pClass1):
  p1 = sum(vex2Classify * p1Vec) + log(pClass1)
  p0 = sum(vex2Classify * p0Vec) + log(1.0 - pClass1)
  if p1 > p0:
    return 1
  else:
    return 0


def testingNB():
  listOPosts, listClasses = loadDataSet()
  myVocabList = createVocabList(listOPosts)
  trainMat = []
  for postingDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList, postingDoc))
  p0v, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
  testEntry = ['love', 'my', 'dalmation']
  thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
  print(f'{testEntry}, classified as: {classifyNB(thisDoc, p0V, p1V, pAb)}')
  testEntry = ['stupid', 'garbage']
  thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
  print(f'{testEntry}, classified as: {classifyNB(thisDoc, p0V, p1V, pAb)}')
