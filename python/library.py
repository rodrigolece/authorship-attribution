import nltk
import os.path
import numpy as np
import scipy.io
import scipy.linalg as la
import itertools
from os import system
from math import inf

text_directory = '../texts'
save_directory = '../mat_files'

def setSaveDirectory(directory):
    global save_directory
    save_directory = directory

# The function words_are not defined globally because they will be attribution dependent
# with open('function_words.txt') as fw:
#     for line in fw:
#         function_words = nltk.word_tokenize(line)

class SampleError(Exception):
    pass

with open('stopper_symbols.txt') as st:
    for line in st:
        stopper_symbols = nltk.word_tokenize(line)

punctuation_symbols = ['.', '?', '!', ';', ',', ':', "''", '``', '--', "(", ")", "'",  "*"]
# punctuation_symbols = stopper_symbols + [',', ':', "''", '``', '--']

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
    def __init__(self, filename, function_words, alpha, D = 10):#, num_words = 10000):
        self.filename = filename
        self.function_words = function_words
        self.alpha = alpha
        self.D = D
        # self.num_words = num_words

        file = os.path.join(text_directory, filename)

        num_sentences = 0

        with open(file) as f:
            all_tokens = nltk.word_tokenize(f.read())
            all_tokens = [token.lower() for token in all_tokens]

            for stopper in stopper_symbols:
                num_sentences += all_tokens.count(stopper)

        # print("Number of sentences parsed: %d" % num_sentences)
        self.all_tokens = all_tokens

    @memoise
    def findFunctionWord(self, word):
        return self.function_words.index(word)

    def idxSubset(self, subset, all_tokens):
          return [i for i, token in enumerate(all_tokens) if token  in subset]

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

    def frequencySubset(self, subset):
        idx = self.idxSubset(subset, self.all_tokens)
        return sum( idx[i] - idx[i-1] for i in range(1,len(idx)) ) / (len(idx) - 1)

    def numWords(self):
        idx_punctuation = self.idxSubset(punctuation_symbols, self.all_tokens)
        return len(self.all_tokens) - len(idx_punctuation)

    # def takeSample(self, num_words):
    #     idx_punctuation = self.idxSubset(punctuation_symbols, self.all_tokens)
    #     no_punctuation = len(idx_punctuation)
    #     density_punctuation = no_punctuation / len(self.all_tokens)
    #     if num_words > len(self.all_tokens) - no_punctuation:
    #         raise SampleError("Sample size is bigger than length of text")
    #
    #     # init_idx = np.random.randint(len(self.all_tokens) - no_punctuation  - num_words)
    #     # this has bias to avoid the end of the text
    #
    #     estimate_punct = int(density_punctuation * num_words)
    #
    #     end_idx = inf;
    #     while end_idx > len(self.all_tokens):
    #         init_idx = np.random.randint(len(self.all_tokens) - int(1.2*estimate_punct)  - num_words)
    #         end_idx = init_idx + num_words + estimate_punct
    #
    #     # Now we correct our estimate
    #     real_punct = len([idx for idx in idx_punctuation if init_idx < idx and idx < end_idx])
    #     current_num_words = end_idx - init_idx - real_punct
    #
    #     if estimate_punct < real_punct:
    #         while current_num_words < num_words:
    #             end_idx += 1
    #             if self.all_tokens[end_idx] not in punctuation_symbols:
    #                 current_num_words += 1
    #     elif estimate_punct > real_punct:
    #         while current_num_words > num_words:
    #             end_idx -= 1
    #             if self.all_tokens[end_idx] not in punctuation_symbols:
    #                 current_num_words -= 1
    #
    #     return (init_idx, end_idx)

    # def sampleCorpus(self, num_words, seed=None):
    #     if seed != None:
    #         np.random.seed(seed)
    #
    #     init_idx = None
    #     num_iters = 0
    #     while init_idx is None and num_iters < 5:
    #         try:
    #             init_idx, end_idx = self.takeSample(num_words)
    #             self.sample = self.all_tokens[init_idx:end_idx]
    #         except SampleError:
    #             raise
    #             break
    #         except ValueError:
    #             num_iters += 1
    #         except IndexError:
    #             num_iters += 1
    #
    #     if num_iters == 5:
    #         raise SampleError("Index ot of range: try with smaller sample")
    #
    # def resetSample(self):
    #     self.sample = None

    def sliceFunctionWords(self, idx_stopper, idx_fwords):
        out = []
        tmp = []
        num_fwords = len(idx_fwords)

        # When we sample we might loose punctuation at end of sentence
        M_stopper, M_fwords = max(idx_stopper), max(idx_fwords)
        if M_stopper < M_fwords:
            idx_stopper.append(M_fwords + 1)

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
                    while current_stopper < fword:
                        counter_stopper += 1
                        current_stopper = idx_stopper[counter_stopper]

            else: # for the last fword do
                tmp.append(fword)
                out.append(tmp)

        return out

    def fillMatrix(self):
        # if self.sample == None:
        #     print("Error: no sample has been defined yet (call ctx.sampleCorpus)")
        #     return None

        idx_stopper = self.idxSubset(stopper_symbols, self.all_tokens) # this gives the indices of the stopper tokens
        idx_fwords = self.idxSubset(self.function_words, self.all_tokens) # same for the function words

        n = len(self.function_words)
        out = np.zeros((n,n))

        fwords_in_sentences = self.sliceFunctionWords(idx_stopper, idx_fwords)

        for sentence in fwords_in_sentences:
            for i in range(len(sentence)-1):
                current_idx = sentence[i]
                next_indices = (idx for idx in sentence[i+1:] if idx - current_idx <= self.D)

                for next_idx in next_indices:
                    d = next_idx - current_idx

                    current_word = self.all_tokens[current_idx]
                    next_word = self.all_tokens[next_idx]

                    current_lin_word = self.findFunctionWord(current_word)
                    next_lin_word = self.findFunctionWord(next_word)

                    out[current_lin_word, next_lin_word] += self.alpha**(d-1)

        return out

    def markovMatrix(self, mat):
        n, _ = mat.shape

        out = mat.copy()

        for row in range(n):
            total = sum(mat[row,:])

            if total == 0:
                out[row,:] = np.ones(n) / n
            else:
                 out[row,:] = out[row,:] / total

        return out

    def buildWAN(self, save_WAN = False, markov = False):
        # if seed != None:
        #     np.random.seed(seed)
        #
        # self.sampleCorpus(num_words)
        out = self.fillMatrix()

        # If we have more than one sample we take an average of all the WANs
        # for i in range(num_samples-1):
        #     self.sampleCorpus(num_words)
        #     out += self.fillMatrix()
        # out /= num_samples

        if markov:
            out = self.markovMatrix(WAN)

        if save_WAN:
            mat_filename = self.filename[:-4] + '.mat'
            full_filename = os.path.join(save_directory, mat_filename)
            # WAN_name = self.filename[:-4] + '_WAN'
            scipy.io.savemat(full_filename, mdict={'WAN': out})
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

def attributionFunction(unknown, candidate_chains):
    no_candidates = len(candidate_chains)
    if no_candidates < 2:
        print("Number of candidates be must at least 2")
        return

    entropies = np.zeros(no_candidates)
    for i, chain in enumerate(candidate_chains):
        entropies[i] = relativeEntropy(unknown, chain)
        print(entropies[i])

    return np.argmin(entropies)

def attributionTest(authors, function_words, alpha, num_words_sample, num_words_corpus, D = 10, N = 10):
    no_candidates = len(authors)
    if no_candidates < 2:
        print("Number of candidates must be at least 2")
        return

    counts = np.zeros(no_candidates)
    WANs = {}

    for candidate in authors[1:]:
      # It would be nice to memoise the context of authors so that we don't build
      # everything again
        ctx = WANcontext(candidate + "_all.txt", function_words, alpha)
        ctx.sampleCorpus(0, num_words_corpus)
        WANs[candidate]= ctx.buildWAN('corpus')

    # The sample always comes from the first author
    main_author = authors[0]
    main_ctx = WANcontext(main_author + "_all.txt", function_words, alpha)

    for i in range(N):
        main_ctx.sampleCorpus(num_words_sample, num_words_corpus)
        WANs[main_author] = main_ctx.buildWAN('corpus')
        sample_WAN = main_ctx.buildWAN('sample')

        chains = [WANs[candidate] for candidate in authors]
        idx = attributionFunction(sample_WAN, chains)
        counts[idx] += 1

        main_ctx.resetSample()

    return (counts[0]/N, counts)


## ---------------------- Corpus tools ---------------------- ##

def buildCorpus(*authors):
    no_authors = len(authors)
    if no_authors == 1:
        author = authors[0]
        cmd = "cat texts/" + author + "_*.txt >> tmp; mv tmp texts/" + author + "_all.txt"
    elif no_authors > 1:
        cmd = ""
        for author in authors:
            cmd += "cat texts/" + author + "_*.txt >> tmp; "
        cmd += "mv tmp texts/test.txt"
    # print(cmd)
    system(cmd)
    # If author does not exist then document is not created



if __name__ == "__main__":
    import sys
    filename = sys.argv[1] # already of type string
    alpha = float(sys.argv[2])

    with open('function_words.txt') as fw:
        for line in fw:
            function_words = nltk.word_tokenize(line)

    ctx = WANcontext(filename, function_words, alpha)
    ctx.buildWAN(save_WAN = True)
