try:
    from .models import LanguageModel, AdjacentLanguageModel, Bigram, load_model
except:
    from models import LanguageModel, AdjacentLanguageModel, Bigram, load_model
try:
    from . import utils
except:
    import utils

def log_likelihood(model: LanguageModel, some_text: str):
    """
    Your code here

    Evaluate the log-likelihood of a given string.

    Hint: utils.one_hot might come in handy

    :param model: A LanguageModel
    :param some_text:
    :return: float
    """

    all_probs = model.predict_all(some_text)
    ll = 0
    for i in range(len(some_text)):
        ll += all_probs[utils.vocab.find(some_text[i]), i]
    return ll


def sample_random(model: LanguageModel, max_length: int = 100):
    import torch
    """
    Your code here.

    Sample a random sentence from the language model.
    Terminate once you reach a period '.'

    :param model: A LanguageModel
    :param max_length: The maximum sentence length
    :return: A string
    """
    res = ''
    while True:
        log_probs = model.predict_next(res)
        probs = torch.exp(log_probs)
        sample = utils.vocab[torch.multinomial(probs, num_samples=1)]
        res += sample
        if sample == '.':
            break
    return res



class TopNHeap:
    """
    A heap that keeps the top N elements around
    h = TopNHeap(2)
    h.add(1)
    h.add(2)
    h.add(3)
    h.add(0)
    print(h.elements)
    > [2,3]

    """
    def __init__(self, N):
        self.elements = []
        self.N = N

    def add(self, e):
        from heapq import heappush, heapreplace
        if len(self.elements) < self.N:
            heappush(self.elements, e)
        elif self.elements[0] < e:
            heapreplace(self.elements, e)

def beam_search(model: LanguageModel, beam_size: int, n_results: int = 10, max_length: int = 100, average_log_likelihood: bool = False, seed_text: str = ''):
    # let the strings terminated with a period persist for comparison at the end, computationally not expensive, efficient for total log likelihood case
    terminated = []
    beams = [(seed_text, 0.0)]  # Start with the seed text and log likelihood of 0
    for _ in range(max_length):
        candidates = []
        for beam, log_likelihood in beams:
            if len(beam) > 0 and beam[-1] == '.':
                continue
            all_probs = model.predict_next(beam)
            new_candidates = [(beam + utils.vocab[i], log_likelihood + all_probs[i]) for i in range(len(utils.vocab))]
            candidates.extend(new_candidates)
            for candidate in new_candidates:
                if candidate[0][-1] == '.':
                    terminated.append(candidate)

        if average_log_likelihood:
            ordered = sorted(candidates, key=lambda x: x[1] / len(x[0]), reverse=True)
        else:
            ordered = sorted(candidates, key=lambda x: x[1], reverse=True)

        beams = ordered[:beam_size]

    beams = list(set(terminated + beams))
    if average_log_likelihood:
        beams = sorted(beams, key=lambda x: x[1] / len(x[0]), reverse=True)
    else:
        beams = sorted(beams, key=lambda x: x[1], reverse=True)

    beams = beams[:n_results]
    res = [beam[0] for beam in beams]
    return res



if __name__ == "__main__":
    """
      Some test code.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', choices=['Adjacent', 'Bigram', 'TCN'], default='Adjacent')
    args = parser.parse_args()

    lm = AdjacentLanguageModel() if args.model == 'Adjacent' else (load_model() if args.model == 'TCN' else Bigram())

    for s in ['abcdefg', 'abcgdef', 'abcbabc', '.abcdef', 'fedcba.']:
        print(s, float(log_likelihood(lm, s)))
    print()

    try:
        for i in range(10):
            s = sample_random(lm)
            print(s, float(log_likelihood(lm, s)) / len(s))
        print()
    except:
        pass

    for s in beam_search(lm, 100, seed_text=''):
        print(s, float(log_likelihood(lm, s)) / len(s))
    print()

    for s in beam_search(lm, 100, average_log_likelihood=True):
        print(s, float(log_likelihood(lm, s)) / len(s))
