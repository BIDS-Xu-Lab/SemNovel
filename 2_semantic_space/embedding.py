import argparse, os, time, joblib, torch, random
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"*** {func.__name__} took {elapsed_time/60:.4f} mins to run.")
        return result
    return wrapper


@timing_decorator
def meta_load(file_path):
    # split the file path into root and extension
    root, extension = os.path.splitext(file_path)
    
    # load df
    if extension == ".tsv":
        df = pd.read_csv(file_path, sep='\t')
    elif extension == ".npy":
        df = np.load(file_path)
    elif extension == ".joblib":
        df = joblib.load(file_path)
    
    # load df info
    print(f'* loaded {len(df)} lines from {file_path}')
    
    if 'year' not in df.columns.tolist():  df = df.rename(columns={'pubdate': 'year'})
    
    # drop na
    columns_to_check = ['pmid', 'year', 'title']
    columns_to_dropna = [c for c in columns_to_check if c in df.columns]
    df = df.dropna(subset=columns_to_dropna).reset_index(drop=True)
    print(f'* found {len(df)} lines after drop na in pmid, year, and title')
    
    # save storage
    # df = df[['pmid','year','journal']]
    df['pmid'] = df.pmid.astype(str)
    df['year'] = df.year.astype(int)
    
    return df


@timing_decorator
def build_col_to_be_embedded(df, cols_to_be_embedded):
    # build col_to_be_embedded
    cols = cols_to_be_embedded.split('&')
    df["col_to_be_embedded"] = df[cols[0]]
    if len(cols) > 1:
        for col in cols[1:]:
            df.col_to_be_embedded = df.apply(lambda r: '%s - %s' % (r.col_to_be_embedded, r[col]), axis=1)
    print('* built col_to_be_embedded')
    return df


@timing_decorator
def calc_BAAI_embeddings(df, model_name):
    # load model
    model = SentenceTransformer(model_name)
    print('* created embedding model: %s' % (model_name))

    # calc embeddings
    print('* start embedding!')
    embeddings = model.encode(
        df.col_to_be_embedded,
        show_progress_bar=True
    )
    print('* done embedding!')
    print(f'* embeddings shape: {embeddings.shape}')
    
    return embeddings



@timing_decorator
def array_save(array, file_path):
    pd.DataFrame(array).to_csv(
        file_path, 
        sep = '\t', 
        index = False
    )
    
    print(f"* saved {array.shape} to {file_path}")
    return array


@timing_decorator
def main():
    parser = argparse.ArgumentParser(description="embedding")
    parser.add_argument("--input_file_path", "-i", type=str, required=True, help="input file path containing the information needed to be embedded")
    parser.add_argument("--cols_to_be_embedded", "-cols", type=str, required=True, help="columns containing the information needed to be embedded, i.e. title&conclusions")
    parser.add_argument("--embedding_method", "-em", choices={"bert","llm"}, default="bert", help="embedding method, i.e. tfidf or bert")
    parser.add_argument("--output_file_path", "-o", type=str, required=True, help="output file path containing the embeddings")
    
    args = parser.parse_args()
    input_file_path = args.input_file_path
    cols_to_be_embedded = args.cols_to_be_embedded
    embedding_method = args.embedding_method
    output_file_path = args.output_file_path

    # data load
    meta_df = meta_load(input_file_path)

    # build col_to_be_embedded
    meta_df = build_col_to_be_embedded(meta_df, cols_to_be_embedded)

    # embedding
    if embedding_method == "bert":
        random.seed(42)
        embeddings = calc_BAAI_embeddings(meta_df, model_name = "BAAI/bge-small-en-v1.5")
    elif embedding_method == "llm":
        random.seed(42)
        embeddings = calc_BAAI_embeddings(meta_df, model_name = "BAAI/llm-embedder")

    # embeddings save
    array_save(embeddings, output_file_path)


if __name__ == "__main__":
    main()