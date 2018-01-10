import re 
import unicodedata
import pickle
import torch
from torch.autograd import Variable
from seq_rnn import EncoderRNN,DecoderRNN
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


def train(lang1,lang2,lang1_n_words,lang2_n_words):
    hidden_size = 256
    encoder = EncoderRNN(lang1_n_words,hidden_size)
    decoder = DecoderRNN(hidden_size,lang2_n_words)
    dataset_size = len(lang1)
    
    # decoder_hidden = decoder.initHidden()
    for i in range(dataset_size):
        encoder_hidden = encoder.initHidden()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input = lang1[i]
        target = lang2[i]
        
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
            decoder_output,decoder_hidden,decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            topv,topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([ni]))
            loss += criterion(decoder_output, target[j])
            if ni == EOS_token:
                break

        break
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

        break
    


eng_lang,fre_lang,eng_lines,fre_lines = prepareData()
print(eng_lines[0], fre_lines[0])

lang1 = getDataset(eng_lang,eng_lines)
lang2 = getDataset(fre_lang,fre_lines)

train(lang1,lang2,eng_lang.n_words,fre_lang.n_words)




        