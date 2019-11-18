'''
Created on Jun 12, 2019

@author: iasl
'''

import sys
import re
import operator
import six
import collections
import math
import tensorflow as tf
import numpy as np
from gensim.models import KeyedVectors
from tensorflow import set_random_seed
from numpy.random import seed
from decimal import Decimal

class SemanticInvariance(object):
    
    def __init__(self, configDesc):
        
        self.invar_Threshold = configDesc['invarThreshold']
        self.InSemCluster = {}
        self.filterSize = configDesc['filterSize']
        self.pretrained_Embedding = KeyedVectors.load_word2vec_format(configDesc['embeddingFile'],binary=True)
        self.featurePath = configDesc['featureFile']
        self.featureEmbed = configDesc['contextEmbedding']
        self.contextEmbedding = {}
        self.contextDimension = 768
        self.embeddingDimension = 300
        self.unseenEmbedding = {}
        self.PosDict ={}
        self.LexDict ={}
        seed(1)
        set_random_seed(2)
        
        if(len(self.pretrained_Embedding.vocab)) == 0:
            tf.logging.info("Error loading the pre-trained embedding")
    
    def retriveSessionValue(self, variableDict):
        
        returnDict = collections.OrderedDict()
        with tf.Session() as sess:
            for index, variableList in variableDict.items():
                #tf.logging.info("%d" %index+'\t %d' %len(variableList))
                returnList = list()
                for variable in variableList:
                    returnList.append(sess.run(variable))
                returnDict.update({index:returnList})
        sess.close()
        return(returnDict)
    
    def __invaraintScore__(self, ts_p20, ts_p11, ts_p02):
        
        return((math.pow(ts_p20, 2))+(math.pow(ts_p11, 2)/2)+(math.pow(ts_p02, 2)))
        
    
    def __invariantClustering__(self, cluster_Tag, semantic_Pattern, invar_Score ):
        
        tier1BufferDict = {}
        if self.InSemCluster.__contains__(cluster_Tag):
            tier1BufferDict = self.InSemCluster.get(cluster_Tag)
        
        tier1BufferList = list()
        if (tier1BufferDict.__contains__(invar_Score)):
            tier1BufferList = tier1BufferDict.get(invar_Score)

        tier1BufferList.append(semantic_Pattern)        
        tier1BufferDict.update({invar_Score : tier1BufferList})
        self.InSemCluster.update({cluster_Tag:tier1BufferDict})
        
        return()
    
    def __invariantOptimization__(self, cluster_Invar, cluster_Tag, orgCluster_Index):
        
        cluster_Dict ={}
        
        def appendEntry(decoyList, value):
            
            if type(value) is list:
                for indexValue in value:
                    if (decoyList.count(indexValue) == 0):
                        decoyList.append(indexValue)
            elif type(value) is float:
                if (decoyList.count(value) == 0):
                    decoyList.append(value)
                
            return(decoyList)
        
        def scanForRangeThreshold(denomValue, append, numValue):
            
            if(append == '9'):
                bufferDenom = (''.join(i for i in denomValue))+str(int(append))
                return(False,list(bufferDenom))
            else:
                bufferDenom = (''.join(i for i in denomValue))+str(int(append)+1)
                maxNum = max([numValue, float(bufferDenom)])
                minDenom = min([numValue, float(bufferDenom)])
                relFraction = str(Decimal(maxNum/minDenom))
                #print(numValue,'\t',bufferDenom,'\t',relFraction,'\t',denomValue)
                relFraction = re.match('.*\..{2}', relFraction).group(0)
                if relFraction == self.invar_Threshold:
                    return(True,list(bufferDenom))
                else:
                    bufferDenom = (''.join(i for i in denomValue))+str(int(append))
                    return(False,list(bufferDenom))
        
        def set_InvarianceRange(decoyList):
            
            tier1BufferResultList = list()
            if(len(decoyList) == 1):
                appendZero = False
                terminalSequence = self.invar_Threshold.split('.')[1]
                tier1BufferList = str(Decimal(decoyList[0])).split('.')
                limitSequence = list(tier1BufferList[0]+'.')
                for index, value in enumerate(list(tier1BufferList[1])):
                    #print(limitSequence)
                    if index <= len(list(terminalSequence)):
                        limitSequence.append(value)
                    else:
                        if(not appendZero):
                            appendZero, limitSequence = scanForRangeThreshold(limitSequence,value,decoyList[0])
                        else:
                            break
                
                maxSeq = float(''.join(entry for entry in limitSequence))
                tier1BufferResultList = [str(decoyList[0]),str(maxSeq)]
            else:
                tier1BufferResultList = [str(min(decoyList)),str(max(decoyList))]
            
            return(tier1BufferResultList)
        
        def clusterIntegration(clusterIndex, decoyList, listCounter):

            #if clusterIndex == 26:
            listCounter = listCounter + len(decoyList)
            tier1BufferResultDict={'is':[],'ip':[]}
            tier1BufferList = list()            
            for (score,patterns) in six.iteritems(cluster_Invar):
                if score in decoyList:
                    tier1BufferList = appendEntry(tier1BufferList, patterns)
            
            # generate range
            updatedList = set_InvarianceRange(decoyList)
            updatedList.sort()
            tier1BufferResultDict.update({'is':updatedList})
            tier1BufferResultDict.update({'ip':tier1BufferList})
            if cluster_Dict.__contains__(clusterIndex):
                tf.logging.info('ENTRY ERROR')
            else:
                cluster_Dict.update({clusterIndex:tier1BufferResultDict})
            
            return(listCounter)
        
        def get_cluster_id(last_index, add_index, new_tag):
            
            mod_val = (last_index%1000)
            last_index = (last_index - mod_val)
            if new_tag:
                current_index = last_index+1000
            else:
                current_index = last_index

            current_index = current_index+add_index
            
            return(current_index)
        
        def generate_context_embedding(sub_cluster_index, invar_range, new_tag):
            
            if self.contextEmbedding:
                last_index = list(self.contextEmbedding.keys())[-1]
                #curr_cluster_index = get_cluster_id(last_index, sub_cluster_index, new_tag)
                curr_cluster_index = last_index+1
            else:
                for curr_cluster_index in range(0,10):
                    invar_distribution = np.random.randn(1, self.contextDimension).reshape(self.contextDimension, )
                    self.contextEmbedding.update({curr_cluster_index:invar_distribution})
                
                last_index = curr_cluster_index
                #curr_cluster_index = get_cluster_id(last_index, sub_cluster_index, new_tag)
                curr_cluster_index = last_index+1
            
            #print('addition ',curr_cluster_index,'\t',sub_cluster_index,'\t',new_tag,'\t',cluster_Tag)
            invar_distribution = np.random.uniform(float(invar_range[0]), float(invar_range[1]), self.contextDimension)
            self.contextEmbedding.update({curr_cluster_index:invar_distribution})
            return()
        
        def writeToFeatureFile():
            
            tier1BufferWriter = open(self.featurePath,"a+")
            if orgCluster_Index == 1:
                tier1BufferWriter = open(self.featurePath,"w+")
                tier1BufferWriter.write('clusterTag\tclusterId\tScore\tPattern\n')

            currentWritePointer = str(cluster_Tag)
            new_tag = True
            for (tier1Key, tier1Value) in six.iteritems(cluster_Dict):
                generate_context_embedding(tier1Key, tier1Value.get('is'), new_tag)
                currentWritePointer = currentWritePointer+"\t"+str(tier1Key)
                currentWritePointer = currentWritePointer+"\t"+str(list(tier1Value.get('is')))
                currentWritePointer = currentWritePointer+"\t"+str(list(tier1Value.get('ip')))+"\n"
                tier1BufferWriter.write(currentWritePointer)
                currentWritePointer=""
                new_tag = False
                tier1BufferWriter.flush()
                
            tier1BufferWriter.close()
                        
            return()
        
        invariantScoreList = list(cluster_Invar.keys())
        preIndex=0
        tier1BufferList = list()
        listCounter = 0
        clusterIndex = 1
        if len(invariantScoreList) == 1:
            preValue = invariantScoreList[preIndex]
            tier1BufferList = appendEntry(tier1BufferList, preValue)
            listCounter = clusterIntegration(clusterIndex, tier1BufferList, listCounter)
            clusterIndex = clusterIndex+1
        else:
            for currIndex in range(1,len(invariantScoreList)):
                preValue = invariantScoreList[preIndex]
                currValue = invariantScoreList[currIndex]
                #print(preValue,'\t',currValue,'\t',clusterIndex)
                tier1BufferList = appendEntry(tier1BufferList, preValue)
                invarianceFraction = str(round(Decimal(preValue/currValue), 2))
                #tf.logging.info('invariance proportion : %s'%invarianceFraction)
                if invarianceFraction == self.invar_Threshold:
                    tier1BufferList = appendEntry(tier1BufferList, currValue)
                else:
                    preIndex = currIndex
                    listCounter = clusterIntegration(clusterIndex, tier1BufferList, listCounter)
                    clusterIndex = clusterIndex+1
                    tier1BufferList = list()
                    
                if(currIndex == len(invariantScoreList)-1):
                    preValue = invariantScoreList[preIndex]
                    tier1BufferList = appendEntry(tier1BufferList, preValue)
                    listCounter = clusterIntegration(clusterIndex, tier1BufferList, listCounter)
                    clusterIndex = clusterIndex+1
                    
        # Sanity Check
        if listCounter != len(set(invariantScoreList)):                
            print('clusterIntegration() ~ Disproportionate group size, counted: %s'%listCounter+" actual: %s"%len(invariantScoreList))
        
        tf.logging.info("%s" %cluster_Tag+'\t Org Sem Cluster: %d'%len(invariantScoreList)+"\t Reduced Sem Cluster: %d"%(clusterIndex-1)+"\n")
        writeToFeatureFile()

        return()
    
    def __geometricInvariance__(self, Pos_Sequence, Lex_Sequence):
        
        
        def recursiveTokenIdentification(currentToken, remainderToken, wordSubTokens):
        
            startIndex = 0
            endIndex = len(currentToken)
            #print("range>>",startIndex,">>>",endIndex,">>",currentToken,"***>>>",remainderToken)
            termIndex = endIndex
            #print("index",termIndex,"token>>",currentToken[termIndex-1])
            bufferToken = currentToken[startIndex:termIndex]
                
            flag = 0
            if bufferToken in self.pretrained_Embedding.vocab:
                dicIndex = len(wordSubTokens)
                wordSubTokens.update({dicIndex:{1:bufferToken}})
                flag=1
                    
            if ((flag == 0) and (termIndex > 1)):
                ''' reducing one letter at a time'''
                remainderToken.append(currentToken[termIndex-1:])
                currentToken = bufferToken[:termIndex-1]
            elif(flag == 1):
                ''' subgroup word structure'''
                if len(remainderToken) > 0:
                    remainderToken.reverse()
                    currentToken = ''.join(charTerm for charTerm in remainderToken)
                    remainderToken = list()
                else:
                    currentToken = None
            else:
                ''' for single words not present with embedding'''
                dicIndex = len(wordSubTokens)
                wordSubTokens.update({dicIndex:{-1:bufferToken}})
                if len(remainderToken) > 0:
                    remainderToken.reverse()
                    currentToken = ''.join(charTerm for charTerm in remainderToken)
                    remainderToken = list()
                else:
                    currentToken = None
                
            if currentToken is not None:
                recursiveTokenIdentification(currentToken, remainderToken, wordSubTokens)
            
            return(wordSubTokens)
        
        def generate_Embedding(tokens, compositeWord):
            #randValue = tf.random_normal(shape=(1,self.embeddingDimension), stddev=0.01)
            #retValue = self.retriveSessionValue(randValue)
            assembledEmbed = np.ones([1, self.embeddingDimension], dtype = np.float32)
            for token in tokens:
                if token in self.pretrained_Embedding.vocab:
                    randValue = np.array([self.pretrained_Embedding.word_vec(token)[0:self.embeddingDimension]])
                    np.reshape(randValue, (1, self.embeddingDimension))
                else:
                    randValue = np.random.rand(1,self.embeddingDimension)
                
                assembledEmbed = np.multiply(assembledEmbed, randValue)
            
            self.unseenEmbedding.update({compositeWord:assembledEmbed})
            
            return()
        
        def retrieve_Embedding(token):
            
            if token in self.pretrained_Embedding.vocab:
                return self.pretrained_Embedding.word_vec(token)[0:self.embeddingDimension]
            else:
                if token not in self.unseenEmbedding:
                    #tf.logging.info(" Tag: %s \n"%token)
                    wordSubTokens = {}
                    wordSubTokens = recursiveTokenIdentification(token,list(), wordSubTokens)
                    generate_Embedding(wordSubTokens, token)
                return self.unseenEmbedding.get(token)[0:self.embeddingDimension]
            
        def generate_components(Pos_Sequence, Lex_Sequence):
            
            contextPosVector = np.ones([1, self.embeddingDimension], dtype=np.float32)
            contextLexVector = np.ones([1, self.embeddingDimension], dtype=np.float32)
            for indexKey, posValue in enumerate(Pos_Sequence):
                lexValue = Lex_Sequence[indexKey]
                tier1NdMatrix = np.array(retrieve_Embedding(posValue), dtype=np.float32).reshape((1, self.embeddingDimension))
                contextPosVector = np.multiply(contextPosVector, tier1NdMatrix)
                
                tier2NdMatrix = np.array(retrieve_Embedding(lexValue), dtype=np.float32).reshape((1, self.embeddingDimension))
                contextLexVector = np.multiply(contextLexVector, tier2NdMatrix)
            
            tensor_initialize = list()
            tensor_initialize.append(np.matmul(contextPosVector, np.reshape(contextPosVector, (self.embeddingDimension, 1))))
            tensor_initialize.append(np.matmul(contextPosVector, np.reshape(contextLexVector, (self.embeddingDimension, 1))))
            tensor_initialize.append(np.matmul(contextLexVector, np.reshape(contextLexVector, (self.embeddingDimension, 1))))
            
            return(tensor_initialize)
        
        def compute_invariance(tensorList):
            
            ts_p20 = tensorList[0][0,0]
            ts_p11 = tensorList[1][0,0]
            ts_p02 = tensorList[2][0,0]
            invar_Score = self.__invaraintScore__(ts_p20, ts_p11, ts_p02)
            #tf.logging.info("(%s)x2" %ts_p20+"+(%s)xy"%ts_p11+"+(%s)y2"%ts_p02+" IScore: %s"%invar_Score)
            
            return(invar_Score)
        
        '''
        tensor_Array = collections.OrderedDict()
        for index in self.PosDict.keys():
            Pos_Sequence = self.PosDict.get(index)
            Lex_Sequence = self.LexDict.get(index) 
            contextPosVector = tf.ones([1, self.embeddingDimension], tf.float32)
            contextLexVector = tf.ones([1, self.embeddingDimension], tf.float32)
            for indexKey, posValue in enumerate(Pos_Sequence):
                lexValue = Lex_Sequence[indexKey]
                tier1NdMatrix = tf.constant(value = retrieve_Embedding(posValue), dtype=tf.float32, shape=(1, self.embeddingDimension))
                contextPosVector = tf.multiply(contextPosVector, tier1NdMatrix)
                
                tier2NdMatrix = tf.constant(value = retrieve_Embedding(lexValue), dtype=tf.float32, shape=(1, self.embeddingDimension))
                contextLexVector = tf.multiply(contextLexVector, tier2NdMatrix)
            
            tensor_initialize = list()
            tensor_initialize.append(tf.matmul(contextPosVector, tf.reshape(contextPosVector, shape = (self.embeddingDimension, 1))))
            tensor_initialize.append(tf.matmul(contextPosVector, tf.reshape(contextLexVector, shape = (self.embeddingDimension, 1))))
            tensor_initialize.append(tf.matmul(contextLexVector, tf.reshape(contextLexVector, shape = (self.embeddingDimension, 1))))
            tensor_Array.update({index:tensor_initialize})
        
        tensor_Array = self.retriveSessionValue(tensor_Array)
        for (key, tensorList) in six.iteritems(tensor_Array):
            ts_p20 = tensorList[0][0,0]
            ts_p11 = tensorList[1][0,0]
            ts_p02 = tensorList[2][0,0]
            invar_Score = self.__invaraintScore__(ts_p20, ts_p11, ts_p02)
            tf.logging.info("%s" %self.PosDict.get(key)+"%s" %self.LexDict.get(key))
            tf.logging.info("(%s)x2" %ts_p20+"+(%s)xy"%ts_p11+"+(%s)y2"%ts_p02+" IScore: %s"%invar_Score)
        '''
        
        tensor_initialize = generate_components(Pos_Sequence, Lex_Sequence)
        invar_Score = compute_invariance(tensor_initialize)
        
        return(invar_Score)
    
    def __unsupervised_clustering__(self):
            
        for index in self.PosDict.keys():
            #tf.logging.info("%s" %self.PosDict.get(index)+"%s" %self.LexDict.get(index))
            Pos_Sequence = self.PosDict.get(index)
            Lex_Sequence = self.LexDict.get(index)
            invar_Score = self.__geometricInvariance__(Pos_Sequence, Lex_Sequence)
            cluster_Tag = self.PosDict.get(index)[0]
            semantic_Pattern = self.PosDict.get(index)
            self.__invariantClustering__(cluster_Tag, semantic_Pattern, invar_Score )
        
            #tensor_Array = self.retriveSessionValue(tensor_Array)
            
        orgCluster_Index = 1
        for (cluster_Tag, cluster_Invar) in six.iteritems(self.InSemCluster):
            #if cluster_Tag == 'TO':
            cluster_Invar = dict(sorted(cluster_Invar.items(), key=operator.itemgetter(0), reverse = True))
            self.__invariantOptimization__(cluster_Invar, cluster_Tag, orgCluster_Index)
            orgCluster_Index = orgCluster_Index+1
            
        tier1BufferWriter = open(self.featureEmbed,"w+")
        for (index_Key, index_Value) in six.iteritems(self.contextEmbedding):
            embed_string = ','.join(str(currVal) for currVal in index_Value)
            currentWritePointer = str(index_Key).strip()+"\t"+str(embed_string).strip()+"\n"
            tier1BufferWriter.write(currentWritePointer)
            currentWritePointer=""
            tier1BufferWriter.flush()

        print("embed length:",len(self.contextEmbedding))                
        tier1BufferWriter.close()
            
        return()
    