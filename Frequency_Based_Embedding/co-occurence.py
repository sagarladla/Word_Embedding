def build_context_window(ctx, st):
    # builds context window of each word in sentence
    word_ctx=dict()
    for i,word in enumerate(st):
        word_ctx[word] = list()
        for left in range(i-ctx,i):
            if left<0:
                continue
            word_ctx[word].append(st[left])
        
        for right in range(i+1,i+(ctx+1)):
            try:
                word_ctx[word].append(st[right])
            except IndexError:
                continue
    return word_ctx

ctx = 2
st = "co-occurence matrix with a fixed context window".split()
print(build_context_window(ctx,st))