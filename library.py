import nltk
import os.path
import numpy as np
import scipy.io
import scipy.linalg as la
import itertools


# The function words_are not defined globally because they will be attribution dependent
# with open('function_words.txt') as fw:
#     for line in fw:
#         function_words = nltk.word_tokenize(line)

with open('stopper_symbols.txt') as st:
    for line in st:
        stopper_symbols = nltk.word_tokenize(line)

def memoise(fun):
  cache = {}
  def memoised_fun(*args, **kwargs):
    kitems = tuple(kwargs.items())
    key = (args, kitems)
    if key not in cache:
      cache[key] = fun(*args, **kwargs)
    return cache[key]
  return memoised_fun

class WANcontext(object):
    def __init__(self, filename, function_words, alpha, D = 10, num_words = 10000):
        self.filename = filename
        self.function_words = function_words
        self.alpha = alpha
        self.D = D
        self.num_words = num_words

        directory = './texts'
        file = os.path.join(directory, filename)

        num_sentences = 0

        with open(file) as f:
            all_tokens = nltk.word_tokenize(f.read())
            all_tokens = [token.lower() for token in all_tokens]

            for stopper in stopper_symbols:
                num_sentences += all_tokens.count(stopper)

        print("Number of sentences parsed: %d" % num_sentences)
        self.all_tokens = all_tokens

    @memoise
    def findFunctionWord(self, word):
        return self.function_words.index(word)

    def findTokens(self, list_tokens):
        return [i for i, token in enumerate(self.all_tokens) if token  in list_tokens]

    def idxMostCommon(self, N):
        counts = np.array([self.all_tokens.count(fword) for fword in self.function_words])
        sort_idx = counts.argsort()[::-1]

        zero_idx = counts[sort_idx].argmin() # since this array is sorted argmin gives the first zero

        if min(N, zero_idx - 1) == N: # zero_idx - 1 is the last function word to appear
            return sort_idx[:N]
        else:
            print("Warning: number of requested function words is more than there are in text")
            print("Returning %d instead" % (zero_idx - 1))
            return sort_idx[:zero_idx-1]


    def sliceFunctionWords(self, idx_stopper, idx_fwords):
        out = []
        tmp = []

        num_fwords = len(idx_fwords)

        counter_stopper = 0
        current_stopper = idx_stopper[counter_stopper]

        for i, fword in enumerate(idx_fwords):
            if i < num_fwords - 1:
                # except for the last fword do
                if fword < current_stopper:
                    tmp.append(fword)
                else:
                    out.append(tmp)
                    tmp = [fword]
                    counter_stopper += 1
                    current_stopper = idx_stopper[counter_stopper]

            else: # for the last fword do
                tmp.append(fword)
                out.append(tmp)

        return out

    def fillMatrix(self):
        n = len(self.function_words)
        out = np.zeros((n,n))

        idx_stopper = self.findTokens(stopper_symbols) # this gives the indices of the stopper tokens in the text
        idx_fwords = self.findTokens(self.function_words) # same for the function words

        fwords_in_sentences = self.sliceFunctionWords(idx_stopper, idx_fwords)

        for sentence in fwords_in_sentences:
            for i in range(len(sentence)-1):
                current_idx = sentence[i]
                next_indices = (idx for idx in sentence[i+1:] if idx - current_idx <= self.D)

                for next_idx in next_indices:
                    d = next_idx - current_idx

                    current_word = self.all_tokens[current_idx]
                    next_word = self.all_tokens[next_idx]
    #                 print(current_word, next_word, d)

                    current_lin_word = self.findFunctionWord(current_word)
                    next_lin_word = self.findFunctionWord(next_word)

                    out[current_lin_word, next_lin_word] += self.alpha**(d-1)

        return out

    def normaliseMatrix(self, mat):
        n,_ = mat.shape

        out = mat.copy()

        for row in range(n):
            total = sum(mat[row,:])

            if total == 0:
                out[row,:] = np.ones(n) / n
            else:
                 out[row,:] = out[row,:] / total

        return out

    def buildWAN(self, save_WAN = False):

        out = self.normaliseMatrix(self.fillMatrix())

        if save_WAN:
            mat_filename = self.filename[:-4] + '.mat'
            WAN_name = self.filename[:-4] + '_WAN'
            scipy.io.savemat(mat_filename, mdict={WAN_name : out})
            return
        else:
            return out


## ---------------------- Entropy stuff ---------------------- ##

def steadyState(mat):
    vals, lvecs, rvecs = la.eig(mat, left=True)

    assert abs(vals[0] - 1.0) < 1e-12

    pi = lvecs[:,0]
    pi = pi.real
    return pi/sum(pi)

def relativeEntropy(chain1, chain2):
    n, _ = chain1.shape
    pi = steadyState(chain1)

#     return sum( pi[i] * chain1[i,j] * np.log(chain1[i,j] / chain2[i,j]) for i, j  in itertools.product(range(n), range(n)) )
# this nice solution brakes down because of division by zero or log(0)

    out = 0.0

    for i, j  in itertools.product(range(n), range(n)):
        if chain1[i,j] == 0.0: # potential trouble here? normally matrix was created using zeros and == should work
            continue
        elif chain2[i,j] == 0.0:
            continue
        else:
            out += pi[i] * chain1[i,j] * np.log(chain1[i,j] / chain2[i,j])

    return out

def attributionFunction(unknown, *candidate_chains):
    no_candidates = len(candidate_chains)
    if no_candidates < 2:
        print("Number of candidates must at least 2")
        return
    else:
        entropies = np.zeros(no_candidates)
        for i, chain in enumerate(candidate_chains):
            entropies[i] = relativeEntropy(chain, unknown)
            print(entropies[i])

        return np.argmin(entropies)



if __name__ == "__main__":
    import sys
    filename = sys.argv[1] # already of type string
    alpha = float(sys.argv[2])

    with open('function_words.txt') as fw:
        for line in fw:
            function_words = nltk.word_tokenize(line)

    ctx = WANcontext(filename, function_words, alpha)
    ctx.buildWAN(save_WAN = True)