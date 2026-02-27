import multiprocessing
import os
import sys
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec, FastText

current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from tokenization.http_tokens import create_request_vector


cores = multiprocessing.cpu_count() 
print(f"Using {cores - 1} out of {cores} cores")

def create_X_embedding_ft(corpus, model=None):
    if not model:
        model = FastText(
                            sentences=corpus,
                            vector_size=100,
                            window=10,
                            min_count=1,
                            sg=1,
                            workers=cores - 1
                        )

    X_vectors = [
        create_request_vector(tokens, model)
        for tokens in tqdm(corpus, desc="Vetorizando")
    ]
    X = np.array(X_vectors)

    return X, model

def create_X_embedding(corpus, model=None):

    if not model:
        model = Word2Vec(
            corpus,
            vector_size=100,
            window=5,
            min_count=5, 
            workers=4
        )

    X_vectors = [create_request_vector(tokens, model) for tokens in corpus]
    X = np.array(X_vectors)

    return X, model
