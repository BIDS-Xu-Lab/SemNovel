import argparse, time, os, joblib
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors
import multiprocessing


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
    df = df[['pmid','year','journal']]
    df['pmid'] = df.pmid.astype(str)
    df['year'] = df.year.astype(int)
    
    return df


@timing_decorator
def array_load(file_path):
    # split the file path into root and extension
    root, extension = os.path.splitext(file_path)

    # load array
    if extension == ".tsv":
        vectors = pd.read_csv(file_path, sep='\t').values
    elif extension == ".npy":
        vectors = np.load(file_path)
    elif extension == ".joblib":
        vectors = joblib.load(file_path)
    
    # load array info
    print(f'* loaded {vectors.shape} vectors from {file_path}')

    # save storage
    memory_usage_before = vectors.nbytes / (1024**3)
    vectors = vectors.astype(np.float32)
    memory_usage_after = vectors.nbytes/ (1024**3)
    print(f"* memory usage of the array improved from {memory_usage_before:.4f} GB to {memory_usage_after:.4f} GB")

    return vectors


def calc_metric_by_index_from_nbrs_k(i, array_year, nbrs_k):
    # get target point
    target_point = array_year[i].reshape(1, -1)
    # calc auc_dk and auc_sk
    try:
        # found k nearest neighbors
        distances, indices = nbrs_k.kneighbors(target_point)    #Euclidean Distance
        auc_dk = sum(distances[0])
    except:
        # not enough k nearest neighbors
        auc_dk = np.nan #Inf
    return auc_dk


def calc_metric_by_year_from_nbrs_k(array, array_year, k):
    # build nbrs for k nearest neighbors
    nbrs_k = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(array)

    # calc metric by year
    metric_list = []
    for i in range(len(array_year)):
        # calc metric by index
        auc_dk = calc_metric_by_index_from_nbrs_k(i, array_year, nbrs_k)
        metric_i = {
            'auc_dk': auc_dk
        }
        metric_list.append(metric_i)
    
    metric = pd.DataFrame(metric_list)
    return metric


def calc_metric_by_year_by_nbrs(year, based_nbrs, shared_dict):
    # get sharing args from manager
    df = shared_dict['df']
    vectors = shared_dict['vectors']
    k = shared_dict['k']
    year_range = shared_dict['year_range']
    tmp_file_folder = shared_dict['tmp_file_folder']

    tmp_file_path = os.path.join(tmp_file_folder, f"{year}_{based_nbrs}.joblib")
    if os.path.exists(tmp_file_path):
        metric = joblib.load(tmp_file_path)
        print(f"* the file {tmp_file_path} exists")
    else:
        # prepare array
        array = vectors[df.year < year]
        array_year = vectors[df.year == year]

        # calc metric by year by based_nbrs
        if array.size == 0 or array_year.size == 0:
            metric = pd.DataFrame()
        else:
            if based_nbrs == 'nbrs_k':
                metric = calc_metric_by_year_from_nbrs_k(array, array_year, k)

        joblib.dump(metric, tmp_file_path)
        print(f"* saved metric to {tmp_file_path}")

    return metric
    

@timing_decorator
def calc_metric(df, vectors, k, year_range, tmp_file_folder):
    # get num_processes
    available_cores = multiprocessing.cpu_count()
    print('* found %s CPU cores' % available_cores)
    num_processes  = max(4, available_cores - 2)
    print('* set num_processes=%s' % num_processes)
    
    # build year list
    min_year = min(df.year)
    max_year = max(df.year)
    year_list = range(min_year+1, max_year+1)
    print(f'* expected {sum(df.year > min_year)} lines with metrics!')
    print(f'* built year_list from {year_list[0]} to {year_list[-1]}')

    # build nbrs list
    nbrs_list = ['nbrs_k']
    print(f'* built nbrs_list: {nbrs_list}')

    # create a manager for sharing args
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()
    shared_dict['df'] = df
    shared_dict['vectors'] = vectors
    shared_dict['k'] = k
    shared_dict['year_range'] = year_range
    shared_dict['tmp_file_folder'] = tmp_file_folder
    print("* created shared_dict")

    with multiprocessing.Pool(processes=num_processes) as pool:
        print('* created a %s-core multiprocessing.Pool' % (num_processes))

        # use the pool to distribute calculating tasks
        args = [(year, based_nbrs, shared_dict) for year in year_list for based_nbrs in nbrs_list]
        print("* created args")

        # get all results
        df_metric_year_nbrs_list = pool.starmap(
                calc_metric_by_year_by_nbrs, 
                args
        )
        
    df_metric = metric_concat(year_list, nbrs_list, tmp_file_folder, df)
    return df_metric


@timing_decorator
def metric_concat(year_list, nbrs_list, tmp_file_folder, df):
    df_metric = pd.DataFrame()
    for year in year_list:
        df_metric_year = df[df.year == year].reset_index(drop=True)
        for based_nbrs in nbrs_list:
            tmp_file_path = os.path.join(tmp_file_folder, f"{year}_{based_nbrs}.joblib")
            df_metric_year_nbrs = joblib.load(tmp_file_path)
            df_metric_year = pd.concat([df_metric_year, df_metric_year_nbrs], axis=1)
        df_metric = pd.concat([df_metric, df_metric_year], axis=0, ignore_index=True)
    
    return df_metric


@timing_decorator
def df_save(df, file_path):
    df.to_csv(
        file_path, 
        sep = '\t', 
        index = False
    )
    
    print(f"* saved {len(df)} lines to {file_path}")
    return df


@timing_decorator
def main():
    parser = argparse.ArgumentParser(description="calculate metrics with sklearn")
    parser.add_argument("--input_file_path", "-i", type=str, required=True, help="input file path containing pmid, year, and journal")
    parser.add_argument("--vectors_file_path", "-vectors", type=str, required=True, help="file path containing the vectors")
    parser.add_argument("--k", "-k", type=int, default=100, help="consider k nearest neighbors")
    parser.add_argument("--year_range", "-yr", type=int, default=10, help="consider year_range when calculating impact metrics")
    parser.add_argument("--tmp_file_folder", "-tmp", type=str, required=True, help="tmp file folder storing intermediate files")
    parser.add_argument("--output_file_path", "-o", type=str, required=True, help="output file path containing metrics")

    args = parser.parse_args()
    input_file_path = args.input_file_path
    vectors_file_path = args.vectors_file_path
    k = args.k
    year_range = args.year_range
    tmp_file_folder = args.tmp_file_folder
    output_file_path = args.output_file_path

    # load data
    df = meta_load(input_file_path)
    # load vectors
    vectors = array_load(vectors_file_path)

    # prepare tmp_file_folder
    if not os.path.exists(tmp_file_folder):
        os.makedirs(tmp_file_folder)

    # calc metrics
    df_metric = calc_metric(df, vectors, k, year_range, tmp_file_folder)
    # save metrics
    df_save(df_metric, output_file_path)
    return df_metric
    

if __name__ == "__main__":
    main()