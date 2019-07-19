from math import log10
from collections import Counter

def unique_token(lit):
    # makes dictionary of unique tokens in whole corpus
    # required in computing Idf
    unique_tokens=list()
    for doc in lit:
        for token in doc.split():
            if token not in unique_tokens:
                unique_tokens.append(token)
    return dict.fromkeys(unique_tokens,0)

def N(lit):
    # counts total docs
    return len(lit)

def n(lit):
    # counts total docs having 'token'
    tokens=unique_token(lit)
    for token in tokens:
        for doc in lit:
            if token in doc.split():
                tokens[token] += 1
                continue
    return tokens

def Idf(lit):
    # idf[token] = log10(count(docs) / count(docs[token]))
    #                    count of total docs / count of docs with token
    #              log10(N/n)
    idf_tokens=n(lit)
    for key,value in idf_tokens.items():
        idf_tokens[key]=log10((N(lit)/value))
    return idf_tokens

def Tf(lit):
    # tf[token] = count(doc[token])/len(doc)
    #            count of token in doc / len of doc
    docs=dict()
    for i in range(N(lit)):
        docs[i]={}
    for key in list(docs.keys()):
        tokens_counts=dict(Counter(lit[key].split()))
        for token,count in tokens_counts.items():
            docs[key][token] = count/len(lit[key].split())
    return docs

def TfIdf(lit):
    tf = Tf(lit)
    idf = Idf(lit)
    tfidf=dict()
    for i in range(len(list(tf.keys()))):
        tfidf[i]={}
    for key in range(len(list(tf.keys()))):
        tokens_tf=tf[key]
        for token,value in tokens_tf.items():
            tfidf[key][token] = value*idf[token]
    return tfidf

def main():
    # corpus containing 2 documents
    lit=["This is about about Messi Messi Messi Messi","This is is about tf-idf"]
    print(TfIdf(lit))

if __name__ == "__main__":
    main()

