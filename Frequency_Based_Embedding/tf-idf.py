from math import log10
from collections import Counter

def unique_token(lit):
    unique_tokens=list()
    for doc in lit:
        for token in doc.split():
            if token not in unique_tokens:
                unique_tokens.append(token)
    return dict.fromkeys(unique_tokens,0)

def N(lit):
    return len(lit)

def n(lit):
    tokens=unique_token(lit)
    for token in tokens:
        for doc in lit:
            if token in doc.split():
                tokens[token] += 1
                continue
    return tokens

def idf(lit):
    idf_tokens=n(lit)
    for key,value in idf_tokens.items():
        idf_tokens[key]=log10((N(lit)/value))
    return idf_tokens

def tf(lit):
    docs=dict()
    for i in range(N(lit)):
        docs[i]={}
    for key in list(docs.keys()):
        tokens_counts=dict(Counter(lit[key].split()))
        for token,count in tokens_counts.items():
            docs[key][token] = count/len(lit[key].split())
    return docs


def main():
    lit=["This is about about Messi Messi Messi Messi","This is is about tf-idf"]
    print(idf(lit))
    print(tf(lit))


if __name__ == "__main__":
    main()

