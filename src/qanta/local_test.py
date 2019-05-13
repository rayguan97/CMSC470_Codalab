from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
from os import path

import numpy as np
import click
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, jsonify, request
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import torch
import torch.nn as nn

# from qanta import util
# from qanta.dataset import QuizBowlDataset


GUESSER_PATH = '../tfidf.pickle'
BUZZ_NUM_GUESSES = 10
BUZZ_THRESHOLD = 0.3
MODEL_PATH = '../buzzer_model_10_.pt'



class RNNBuzzer(nn.Module):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    We use a LSTM for our buzzer.
    """

    #### You don't need to change the parameters for the model

    #n_input represents dimensionality of each specific feature vector, n_output is number of labels
    def __init__(self, n_input=10, n_hidden=50, n_output=2, dropout=0.5):
        super(RNNBuzzer, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.n_output = n_output
        
        ### Your Code Here --- 
        #define lstm layer, going from input to hidden. Remember to have batch_first=True.
        self.lstm = nn.LSTM(input_size=self.n_input, hidden_size=self.n_hidden, batch_first=True) ##
        
        #define linear layer going from hidden to output.
        self.hidden_to_label = nn.Linear(self.n_hidden, self.n_output) ##
      
        
        #---you can add other things like dropout between two layers, but do so in forward function below,
        #as we have to perform an extra step on the output of LSTM before going to linear layer(s).
        #The model for test cases is just single layer LSTM followed by 1 linear layer - do not add anything else for the purpose of passing test cases!!
    def forward(self, X, X_lens):
        """
        Model forward pass, returns the logits of the predictions.
        
        Keyword arguments:
        input_text : vectorized question text 
        text_len : batch * 1, text length for each question
        is_prob: if True, output the softmax of last layer
        """
        
        #get the batch size and sequence length (max length of the batch)
        #dim of X: batch_size x batch_max_len x input feature vec dim
        batch_size, seq_len, _ = X.size()
        
        ###Your code here --
        
        #Get the output of LSTM - (output dim: batch_size x batch_max_len x lstm_hidden_dim)
        temp, _ = self.lstm(X)  ##

        
        #reshape (before passing to linear layer) so that each row contains one token 
        #essentially, flatten the output of LSTM 
        #dim will become batch_size*batch_max_len x lstm_hidden_dim
        temp = temp.contiguous() ##
        temp = temp.view(-1, temp.shape[2]) ##
        
        #Get logits from the final linear layer
        logits = self.hidden_to_label(temp) ##
        
        #--shape of logits -> (batch_size, seq_len, self.n_output)
        return logits



def guess_and_buzz(model, guesser, question_text) -> Tuple[str, bool]:
    # here we pass in a list of strings
    # we will get a list(number of strings) of 
    # list (number of gueeses) of tutples
    guesses = guesser.guess([question_text], BUZZ_NUM_GUESSES)

    scores = [guess[1] for guess in guesses[0]]
    buzz = int(model(torch.tensor([[scores]]), 1).topk(1)[1]) == 1
    return guesses[0][0][0], buzz


def batch_guess_and_buzz(model, guesser, questions) -> List[Tuple[str, bool]]:
    question_guesses = guesser.guess(questions, BUZZ_NUM_GUESSES)
    outputs = []
    for guesses in question_guesses:
        scores = [guess[1] for guess in guesses]

        buzz = int(model(torch.tensor([[scores]]), 1).topk(1)[1]) == 1

        outputs.append((guesses[0][0], buzz))
    return outputs


class TfidfGuesser:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.i_to_ans = None

    def train(self, training_data) -> None:
        questions = training_data[0]
        answers = training_data[1]
        answer_docs = defaultdict(str)
        for q, ans in zip(questions, answers):
            text = ' '.join(q)
            answer_docs[ans] += ' ' + text

        x_array = []
        y_array = []
        for ans, doc in answer_docs.items():
            x_array.append(doc)
            y_array.append(ans)

        self.i_to_ans = {i: ans for i, ans in enumerate(y_array)}
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3), min_df=2, max_df=.9
        ).fit(x_array)
        self.tfidf_matrix = self.tfidf_vectorizer.transform(x_array)

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        representations = self.tfidf_vectorizer.transform(questions)
        guess_matrix = self.tfidf_matrix.dot(representations.T).T
        guess_indices = (-guess_matrix).toarray().argsort(axis=1)[:, 0:max_n_guesses]
        guesses = []
        for i in range(len(questions)):
            idxs = guess_indices[i]
            guesses.append([(self.i_to_ans[j], guess_matrix[i, j]) for j in idxs])

        return guesses

    def save(self):
        with open(GUESSER_PATH, 'wb') as f:
            pickle.dump({
                'i_to_ans': self.i_to_ans,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix
            }, f)

    @classmethod
    def load(cls):
        with open(GUESSER_PATH, 'rb') as f:
            params = pickle.load(f)
            guesser = TfidfGuesser()
            guesser.tfidf_vectorizer = params['tfidf_vectorizer']
            guesser.tfidf_matrix = params['tfidf_matrix']
            guesser.i_to_ans = params['i_to_ans']
            return guesser




if __name__ == '__main__':
    tfidf_guesser = TfidfGuesser.load()
    model = torch.load(MODEL_PATH)
    test1 = ['name this ',
         'name this first unit',
         'name this first united state p',
         'name this first united state president.']
    test2 =[['name this '],
         ['name this first unit'],
         ['name this first united state p'],
         ['name this first united state president.'] ]
    # correct args: test1 or test2[1]
    test3 = []

    # scores = [guess[1] for guess in guesses[0]]
    # print(guesses)
    # print(np.shape(guesses))
    # print(scores)
    # # print(torch.tensor([[scores]]))
    # # print(np.shape(torch.tensor([[scores]])))



    # res = int(model(torch.tensor([[scores]]), 1).topk(1)[1]) == 1

    # print(res)
    # print(guesses[0][0][0])


    # guesses = tfidf_guesser.guess(test1, BUZZ_NUM_GUESSES)

    # outputs = []
    # for guess in guesses:
    #     scores = [g[1] for g in guess]

    #     buzz = int(model(torch.tensor([[scores]]), 1).topk(1)[1]) == 1

    #     outputs.append((guesses[0][0][0], buzz))
    # print(outputs)


    print(guess_and_buzz(model, tfidf_guesser, test1[3]))
    print(batch_guess_and_buzz(model, tfidf_guesser, test1))



