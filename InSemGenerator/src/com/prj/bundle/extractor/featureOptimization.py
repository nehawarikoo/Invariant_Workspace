'''
Created on Jun 11, 2019

@author: iasl
'''
import sys
import os
import re
import json
import six
import collections
import tensorflow as tf
import socket

if socket.gethostname() == 'iaslgpu3':
    sys.path.append('/home/neha/nlp/NeonWorkspace_1.6/InSemGenerator')
elif  socket.gethostname() == 'iaslgpu5':
    sys.path.append('/home/iasl/Neha_W/NeonWorkspace_1.6/InSemGenerator')


import src.com.prj.bundle.extractor.featureInvariance as featureInvariance

class InputDataStream(object):
    
    def __init__(self):
        self.configDesc = {}
        self.PostTagStream = {}
        self.LexicalStream = {}
        self.InSemCluster = {}
        
    def featureConvolution(self):
        
        semInvariance = featureInvariance.SemanticInvariance(self.configDesc)
        semInvariance.PosDict = collections.OrderedDict()
        semInvariance.LexDict = collections.OrderedDict()
        filterSize = int(self.configDesc.get("filterSize"))
        strideSize = int(self.configDesc.get("strideSize"))
        
        patternIndex = 0
        for tier1Key, tier1Value in self.PostTagStream.items():
            tier2Value = self.LexicalStream.get(tier1Key)
            # Santity Check
            if len(tier1Value) != len(tier2Value):
                tf.logging.info("critical error with pre-processing %s ~ featureConvolution()" % tier1Key)
                tf.logging.info("%s " % len(tier1Value) +" %s" % len(tier2Value))
            
            startIndex = 0
            endIndex = filterSize
            max_SequenceLength = len(tier1Value)
            while (startIndex < (max_SequenceLength-filterSize+1)) and (endIndex < max_SequenceLength):
                endIndex = (startIndex+filterSize)
                tier1BufferResultList = list(tier1Value[startIndex:endIndex])
                tier2BufferResultList = list(tier2Value[startIndex:endIndex])
                if(tier1BufferResultList.count('#') == filterSize):
                    break
                else:
                    semInvariance.PosDict.update({patternIndex:tier1BufferResultList})
                    semInvariance.LexDict.update({patternIndex:tier2BufferResultList})
                    patternIndex = patternIndex+1
                    #tf.logging.info("%s" %tier1BufferResultList+"\t %s" %tier2BufferResultList)
                    startIndex = startIndex+strideSize

        tf.logging.info("feature depth %d" %len(semInvariance.PosDict))
        semInvariance.__unsupervised_clustering__()

        return()
    
    def sequencePadding(self, maxTokenLength):
        
        strideSize = int(self.configDesc.get("strideSize"))
        addLength = maxTokenLength%strideSize
        if addLength > 0:
            maxTokenLength = maxTokenLength+addLength
            
        for tier1Key, tier1Value in self.PostTagStream.items():
            tier2Value = self.LexicalStream.get(tier1Key)
            for tier1Index in range(maxTokenLength-len(tier1Value)):
                tier1Value.append('#')
                tier2Value.append('#')
            self.PostTagStream.update({tier1Key:tier1Value})
            self.LexicalStream.update({tier1Key:tier2Value})
        
        return()
    
    def initializeCluster(self, decoySet):
        
        for indexValue in decoySet:
            self.InSemCluster.update({indexValue:{'Pos':{},'Lex':{}}})
        
        return()
    
    def readInputStream(self):
        
        maxTokenLength = 0
        self.PostTagStream = collections.OrderedDict()
        self.LexicalStream = collections.OrderedDict()
        tier0BufferList = list()
        with open(self.configDesc["posTagFile"], "r") as bufferFile:
            currentData = bufferFile.readline()
            while len(currentData)!=0:
                tier1BufferList = list(str(currentData).split(sep='\t'))
                instanceId = str(tier1BufferList[0]).strip()
                instanceText = list(str(tier1BufferList[1]).strip().rsplit(sep=' '))
                self.PostTagStream.update({instanceId:instanceText})
                tier0BufferList.extend(instanceText)
                if maxTokenLength < len(instanceText):
                    maxTokenLength = len(instanceText)
                currentData = bufferFile.readline()
        bufferFile.close()
        self.initializeCluster(set(tier0BufferList))
        
        with open(self.configDesc["lexicalFile"], "r") as bufferFile:
            currentData = bufferFile.readline()
            while len(currentData)!=0:
                tier1BufferList = list(str(currentData).split(sep='\t'))
                instanceId = str(tier1BufferList[0]).strip()
                instanceText = list(str(tier1BufferList[1]).strip().rsplit(sep=' '))
                self.LexicalStream.update({instanceId:instanceText})
                currentData = bufferFile.readline()
        bufferFile.close()

        self.sequencePadding(maxTokenLength)
        self.featureConvolution()        
        return()

def openConfigurationFile(decoyInstance):
    
    path = os.getcwd()
    tokenMatcher = re.search(".*InSemGenerator\/", path)
    configFile = ""
    if tokenMatcher:
        configFile = tokenMatcher.group(0)
        configFile="".join([configFile,"config.json"])
        
    #with tf.gfile.GFile(configFile, "r") as reader:
    tier1BufferDict = {}
    with open(configFile, "r") as json_file:
        data = json.load(json_file)
        for (key, value) in six.iteritems(data):
            tier1BufferDict.update({key:value})
        json_file.close()
    decoyInstance.configDesc = tier1BufferDict
            
    return()

def main(_):
    
    inputStreamInstance = InputDataStream()
    tf.logging.set_verbosity(tf.logging.INFO)
    openConfigurationFile(inputStreamInstance)
    inputStreamInstance.readInputStream()
    
if __name__ == "__main__":
    tf.app.run()