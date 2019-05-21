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

from qanta import getWikiData
from qanta import util
from qanta.dataset import QuizBowlDataset

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import qanta.LSTM as LSTM


MODEL_PATH = 'tfidf.pickle'
BUZZ_NUM_GUESSES = 10
BUZZ_THRESHOLD = 0.3
CHAR_SKIP = 50


def guess_and_buzz(lstm, model, question_text) -> Tuple[str, bool]:
     
    #Get a single question

    guesses = model.guess([question_text], BUZZ_NUM_GUESSES)[0]
     
    scores = [guess[1] for guess in guesses] #The scores from all the 10 guesses
    

    #print(lstm(torch.tensor([[scores]]), 1).topk(1)[1])
    buzz = int(lstm(torch.tensor([[scores]]), 1).topk(1)[1]) == 1
    #buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD #Is first one greater than other
    return guesses[0][0], buzz #Return name of highest guess and buzz result


def batch_guess_and_buzz(lstm, model, questions) -> List[Tuple[str, bool]]:
    #Get a list of questions
    question_guesses = model.guess(questions, BUZZ_NUM_GUESSES)

    outputs = []
    for guesses in question_guesses:
        scores = [guess[1] for guess in guesses]
        #buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD

        buzz = int(lstm(torch.tensor([[scores]]), 1).topk(1)[1]) == 1
        outputs.append((guesses[0][0], buzz))
    return outputs


class TfidfGuesser:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.i_to_ans = None
        self.neural_model = None

    def train(self, training_data, wiki_x, wiki_y) -> None:

        questions, answers = [], []
        for ques in training_data:
            questions.append(ques.sentences)
            answers.append(ques.page)
        
        answer_docs = defaultdict(str)

        for q, ans in zip(questions, answers):
            text = ' '.join(q) #Joins all sentences of a question
            answer_docs[ans] += ' ' + text

        # for q, ans in zip(wiki_x, wiki_y):
        #     text = ' '.join(q) #Joins all sentences of a question
        #     answer_docs[ans] += ' ' + text

        x_array = []
        y_array = []
        for ans, doc in answer_docs.items():
            x_array.append(doc)
            y_array.append(ans)


        self.i_to_ans = {i: ans for i, ans in enumerate(y_array)}
        
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 5), min_df=4, max_df=.7
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
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({
                'i_to_ans': self.i_to_ans,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix
            }, f)

    @classmethod
    def load(cls):
        with open(MODEL_PATH, 'rb') as f:
            params = pickle.load(f)
            guesser = TfidfGuesser()
            guesser.tfidf_vectorizer = params['tfidf_vectorizer']
            guesser.tfidf_matrix = params['tfidf_matrix']
            guesser.i_to_ans = params['i_to_ans']
            return guesser
################

###################

def create_app(enable_batch=True): #get loaded data
    tfidf_guesser = TfidfGuesser.load()
    #load LSTM
    lstm = LSTM.RNNBuzzer()
    lstm.load_state_dict(torch.load('lstm_model.pt'))
    lstm.eval()

    print("Models loaded")


    app = Flask(__name__)

    @app.route('/api/1.0/quizbowl/act', methods=['POST'])
    def act():
         
        question = request.json['text']
        guess, buzz = guess_and_buzz(lstm, tfidf_guesser, question)
        return jsonify({'guess': guess, 'buzz': True if buzz else False})

    @app.route('/api/1.0/quizbowl/status', methods=['GET'])
    def status():
         
        return jsonify({
            'batch': enable_batch,
            'batch_size': 200,
            'ready': True,
            'include_wiki_paragraphs': False
        })

    @app.route('/api/1.0/quizbowl/batch_act', methods=['POST'])
    def batch_act():
        
        questions = [q['text'] for q in request.json['questions']]
         
        return jsonify([
            {'guess': guess, 'buzz': True if buzz else False}
            for guess, buzz in batch_guess_and_buzz(lstm, tfidf_guesser, questions)
        ])


    return app


@click.group()
def cli():
    pass


@cli.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=4861)
@click.option('--disable-batch', default=False, is_flag=True)
def web(host, port, disable_batch):
    """
    Start web server wrapping tfidf model
    """
    app = create_app(enable_batch=not disable_batch) #Disable or not
    app.run(host=host, port=port, debug=False)


@cli.command()
def train():
    """
    Train the tfidf model, requires downloaded data and saves to models/
    """
    print("At train 1 : Getting Datasets")
    
    #See if file exist already
    exists = path.isfile('tfidf.pickle') and path.isfile('train_exs.npy') and path.isfile('dev_exs.npy')
    if not exists:
        datasetTrainGuess = QuizBowlDataset(guesser_train=True).training_data()
        datasetTrainBuzz = QuizBowlDataset(buzzer_train=True).training_data()
        datasetDevBuzz = QuizBowlDataset(buzzer_dev=True).training_data()

        

    #Download Wikipedia data if not found
    exists = path.isfile('wikidata.json')
    if not exists:
        print('creating wiki data file')
        getWikiData.get_wikipedia_data()



    # datasetTrainGuess[0] -- questions
    #    #Each question has its own set of setences.
    # datasetTrainGuess[1] -- answers    

    # Check if pickle exists
    print("At train 2 : Getting Guessers and train them")
    exists = path.isfile('tfidf.pickle') 
    if exists:
        tfidf_guesser = TfidfGuesser.load()
    else:
        print('loading wiki')
        with open('wikidata.json') as f:
            data = json.load(f)

        print('Finish loading')
        questions_wiki = []
        answers_wiki = []
        i = 0
        for val in data:
            i += 1
            if i > 500:
                break;
            answers_wiki.append(data[val]['title'])
            questions_wiki.append(data[val]['text'])

        print("training data ready")
        tfidf_guesser = TfidfGuesser()
        tfidf_guesser.train(datasetTrainGuess, questions_wiki, answers_wiki)
        tfidf_guesser.save()

    

    #Check if trained examples exists

    print("At train 3 : Train them")
    exists = path.isfile('train_exs.npy') and path.isfile('dev_exs.npy')
    if exists:
        train_exs = np.load('train_exs.npy')
        dev_exs = np.load('dev_exs.npy')
    else:
        #Phrase skip should go in the function under here
        print('Generating Guesses for Training Buzzer Data')
        train_qnums, train_answers, train_char_indices, train_ques_texts, train_ques_lens, train_guesses_and_scores =  \
                                                                    LSTM.generate_guesses_and_scores(tfidf_guesser, datasetTrainBuzz, BUZZ_NUM_GUESSES, char_skip=CHAR_SKIP)
      
        print('Generating Guesses for Dev Buzzer Data')
        dev_qnums, dev_answers, dev_char_indices, dev_ques_texts, dev_ques_lens, dev_guesses_and_scores =    \
                                                                LSTM.generate_guesses_and_scores(tfidf_guesser, datasetDevBuzz, BUZZ_NUM_GUESSES, char_skip=CHAR_SKIP)
        

        train_exs = LSTM.create_feature_vecs_and_labels(train_guesses_and_scores, train_answers, BUZZ_NUM_GUESSES)   
        dev_exs = LSTM.create_feature_vecs_and_labels(dev_guesses_and_scores, dev_answers, BUZZ_NUM_GUESSES)
        np.save('train_exs.npy', train_exs)
        np.save('dev_exs.npy', dev_exs)

    # Organize data
    print("At train 4: Data")
    train_dataset = LSTM.QuestionDataset(train_exs)
    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=8, sampler=train_sampler, num_workers=0,
                                           collate_fn=LSTM.batchify)

    dev_dataset = LSTM.QuestionDataset(dev_exs)
    dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
    dev_loader = DataLoader(dev_dataset, batch_size=8, sampler=dev_sampler, num_workers=0,
                                           collate_fn=LSTM.batchify)

    print("At train 5: Train LSTM")
    lstm = LSTM.RNNBuzzer()


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lstm.to(device)
    for epoch in range(25):
        print('start epoch %d' %(epoch+1))
        train_acc, dev_acc = LSTM.train(50, lstm, train_loader, dev_loader, device)


    print("At train 6: Saving LSTM model")

    torch.save(lstm.state_dict(), 'lstm_model.pt')

    print("Done with training")




@cli.command()
@click.option('--local-qanta-prefix', default='data/')
@click.option('--retrieve-paragraphs', default=False, is_flag=True)
def download(local_qanta_prefix, retrieve_paragraphs):
    """
    Run once to download qanta data to data/. Runs inside the docker container, but results save to host machine
    """
    #print("\n\n\tLocal  Qanta Prefix: %s" %local_qanta_prefix)
    #print("\n\n\tRetrieve Paragraphs: %s" %retrieve_paragraphs)
    

    util.download(local_qanta_prefix, retrieve_paragraphs)


if __name__ == '__main__':
    cli()