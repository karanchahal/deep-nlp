import re 
import unicodedata
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from seq_rnn import EncoderRNN,DecoderRNN,AttentionDecoderRNN
import torch.optim as optim
DATASET_PATH = './data/eng-fra.txt'

MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

class Lang:
    def __init__(self,name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"SOS",1:"EOS"}
        self.n_words = 2
    
    def addSentence(self,sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    
    def addWord(self,word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def preprocessString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub("([.!?])","\1",s)
    s = re.sub("[^a-zA-Z.!?]+", " ", s)
    return s

def readLines():
    french_lines = []
    eng_lines = []

    lines = open(DATASET_PATH).read().strip().split('\n')

    for li in lines:
        pairs = li.split('\t')
        eng_lines.append(preprocessString(pairs[0]))
        french_lines.append(preprocessString(pairs[1]))
    
    print(french_lines[0])
    print(eng_lines[0])
    pickle.dump(french_lines,open('data/fre.p','wb'))
    pickle.dump(eng_lines,open('data/eng.p','wb'))
    return eng_lines,french_lines

def loadLines():
    eng_lines = pickle.load(open('data/eng.p','rb'))
    fre_lines = pickle.load(open('data/fre.p','rb'))

    return eng_lines,fre_lines


def filter(p1,p2):
    return len(p1.split(' ')) < MAX_LENGTH and \
        len(p2.split(' ')) < MAX_LENGTH and \
        p1.startswith(eng_prefixes)

def filterExamples(eng_lines,fre_lines):
    filtered_eng = []
    filtered_fre = []
    for i in range(len(eng_lines)):
        if(filter(eng_lines[i],fre_lines[i])):
            filtered_eng.append(eng_lines[i])
            filtered_fre.append(fre_lines[i])
    
    return filtered_eng,filtered_fre

def prepareData():

    eng_lines,fre_lines = loadLines()
    eng_lines,fre_lines = filterExamples(eng_lines,fre_lines)

    eng_lang = Lang('eng')
    fre_lang = Lang('fre')

    for a,b in zip(eng_lines,fre_lines):
        eng_lang.addSentence(a)
        fre_lang.addSentence(b)
    return eng_lang,fre_lang,eng_lines,fre_lines

# we send list of indexes into the model and output list of indexes
def linetoIndexes(lang,line):
    indexes = []
    for word in line.split(' '):
        if word != '':
            indexes.append(lang.word2index[word])

    return indexes

def getDataset(lang,lines):
    dataset = []
    for li in lines:

        data = linetoIndexes(lang,li)
        data.append(EOS_token)
        dataset.append(Variable(torch.LongTensor(data)))

   
    return dataset


def train(lang1,lang2,lang1_n_words,lang2_n_words,print_every=1, plot_every=100):
    hidden_size = 256
    learning_rate = 0.01
    dataset_size = len(lang1)

    encoder = EncoderRNN(lang1_n_words,hidden_size)
    decoder = AttentionDecoderRNN(hidden_size,lang2_n_words)
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    print_loss_total = 0
    plot_loss_total = 0
    start = time.time()
    plot_losses = []
    
    # decoder_hidden = decoder.initHidden()
    for i in range(1,dataset_size+1):
        encoder_hidden = encoder.initHidden()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss = 0

        input = lang1[i-1]
        target = lang2[i-1]
        
        input_len = input.size(0)
        target_len = target.size(0)

        encoder_outputs = Variable(torch.zeros(MAX_LENGTH,encoder.hidden_size))
        # encoder
        for j in range(input_len):
            encoder_output,encoder_hidden = encoder(input[j],encoder_hidden)
            encoder_outputs[j] = encoder_output[0][0]
        
        decoder_input = Variable(torch.LongTensor([SOS_token]))
        decoder_hidden = encoder_hidden

        for j in range(target_len):
            decoder_output,decoder_hidden,decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv,topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([ni]))
            loss += criterion(decoder_output, target[j])
            if ni == EOS_token:
                break
   
        loss.backward()
        loss = loss.data[0]/target_len

        encoder_optimizer.step()
        decoder_optimizer.step()

        print_loss_total += loss
        plot_loss_total += loss

        if i% print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, i/ dataset_size),
                                         i, i / dataset_size * 100, print_loss_avg))
        if i % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        
    showPlot(plot_losses)

    


eng_lang,fre_lang,eng_lines,fre_lines = prepareData()
print(eng_lines[0], fre_lines[0])

lang1 = getDataset(eng_lang,eng_lines)
lang2 = getDataset(fre_lang,fre_lines)

train(lang1,lang2,eng_lang.n_words,fre_lang.n_words)




        