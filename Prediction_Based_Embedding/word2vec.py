import numpy as np
import utils

class Word2Vec:
    def __init__(self, **kwargs):
        self.ctx = kwargs['ctx']
        self.corpus = kwargs['corpus']
        self.vocab = self.setvocab()
        self.lenvocab = len(self.vocab)
        self.dim = kwargs['dim']
        self.epochs = kwargs['epochs']
        self.alpha = kwargs['alpha']
        self.word_index, self.index_word = self.lookupdict()
        if self.dim>self.lenvocab:
            raise(utils.DimensionException("Hidden layer neurons should be less or equal to length of vocabulary"))
        self.em = self.embeddingmatrix()
        self.cm = self.contextmatrix()

    def generatetrainingdata(self):
        trainingdata=dict()
        encodedvocab = self.onehotenc()
        contextdata = self.buildcontext()
        for key, value in contextdata.items():
            contextvalues=list()
            for item in value:
                contextvalues.append(np.array(encodedvocab[item]))
            trainingdata[encodedvocab[key]]=np.array(contextvalues)
        return trainingdata

    def lookupdict(self):
        word_index=dict()
        index_word=dict()
        encodedvocab=self.onehotenc()
        for word, enc in encodedvocab.items():
            for n,val in enumerate(enc):
                if val==1:
                    word_index[word] = n
                    index_word[n] = word
        return word_index, index_word

    def setvocab(self):
        vocab=set()
        for st in self.corpus:
            for word in st.split():
                vocab.add(word)
        return sorted(vocab)

    def onehotenc(self):
        wordenc=dict()
        for wordi in self.vocab:
            enc=list()
            for wordj in self.vocab:
                if wordi==wordj:
                    enc.append(1)
                else:
                    enc.append(0)
            wordenc[wordi]=tuple(enc)
        return wordenc

    def buildcontext(self):
        word_ctx=dict()
        for st in self.corpus:
            for word in st.split()  :
                word_ctx[word]=list()
        for st in self.corpus:
            st=st.split()
            for i,word in enumerate(st):
                for left in range(i-self.ctx,i):
                    if left<0:
                        continue
                    word_ctx[word].append(st[left])
                
                for right in range(i+1,i+(self.ctx+1)):
                    try:
                        word_ctx[word].append(st[right])
                    except IndexError:
                        continue
        return word_ctx
    
    def embeddingmatrix(self):
        return np.random.uniform(-0.8,0.8,(self.lenvocab,self.dim))
    
    def contextmatrix(self):
        return np.random.uniform(-0.8,0.8,(self.dim,self.lenvocab))
    
    def softmax(self,x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    def forwardpass(self, ctxword):
        h = np.dot(self.em.T,ctxword)
        u = np.dot(self.cm.T, h)
        y = self.softmax(u)
        return (h,u,y)

    def error(self,ctx_words, y_pred):
        err_list=list()
        for word in ctx_words:
            err_list.append(np.subtract(y_pred,word))
        total_error = np.sum(err_list, axis=0)
        return total_error

    def backpropagation(self, error,hidden_weights,embedded_weights):
        d_cm = np.outer(hidden_weights, error)
        d_em = np.outer(embedded_weights, np.dot(self.cm,error.T))
        self.em = self.em - (self.alpha * d_em)
        self.cm = self.cm - (self.alpha * d_cm)

    def calculateloss(self, context_words, u):
        templist=list()
        for ctxword in context_words:
            templist.append(u[ctxword.index(1)])
        loss = -np.sum(templist)+len(context_words)*np.log(np.sum(np.exp(u)))
        return loss

    def train(self):
        for i in range(self.epochs):
            self.loss=0
            for target_word, context_words in self.generatetrainingdata().items():
                target_word = np.array(target_word)
                h,u,y_pred = self.forwardpass(target_word)
                Error = self.error(context_words, y_pred)
                self.backpropagation(Error, h, target_word)
                # self.loss += self.calculateloss(context_words,u)
            print("EPOCH: ",i)#, "LOSS: ",self.loss)
    
    def wordsimilarity(self, word):
        wi = self.word_index[word]
        weightmat_word = self.em[wi]
        wordsim=dict()
        for i in range(self.lenvocab):
            weightmat_ctx = self.em[i]
            tn = np.dot(weightmat_word,weightmat_ctx)
            td = np.linalg.norm(weightmat_word)*np.linalg.norm(weightmat_ctx)
            t=tn/td
            word=self.index_word[i]
            wordsim[word]=t
        wordsim = sorted(wordsim.items(), key=lambda v :v[1], reverse=True)
        return wordsim

def main():
    corpus = [
                "લોકો પોતાના સ્વાસ્થ્ય માટે ખૂબ સજાગ છે તેથી જ ઘરમાં બનતી રસોઈમાં કઈ કઈ વસ્તુઓનો ઉપયોગ થશે તે વાત પણ પરીવાર માટે મહત્વની હોય છે",
                "મોટાભાગના પરીવાર પોતાના ઘરમાં રસોઈ માટે ઓલિવ ઓઈલ કે ખાસ પ્રકારના તેલનો જ ઉપયોગ કરવાનો આગ્રહ રાખે છે"
            ]
    w2v = Word2Vec(ctx=2,dim=3,corpus=corpus,epochs=5000,alpha=0.05)
    # print(w2v.vocab)
    # print(w2v.buildcontext())
    # train_data=w2v.generatetrainingdata()
    # for wt, wc in train_data.items():
    #    h,u,y=w2v.forwardpass(wt)
    #    print("h:",h)
    #    print("u:",u)
    #    print("y:",y)
    #    print("error:",w2v.error(wc,y))
    # print(w2v.em)
    # print(w2v.cm)
    w2v.train()
    print(w2v.wordsimilarity("પરીવાર"))
    # print(w2v.em)
    # print(w2v.cm)

if ___name__ == '__main__':
    main()
