import argparse, time, random, subprocess, multiprocessing, tempfile
import pandas as pd
import numpy as np
from openTSNE import TSNE


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
def calc_tsne(embeddings):
    # create tsne object
    tsne = TSNE(
        perplexity=30,
        metric="cosine",
        n_jobs=4,
        random_state=42,
        verbose=True,
    )
    print('* created TSNE object')

    # tsne it!
    print('* start tsne!')
    embds = tsne.fit(embeddings[:, :])
    print('* done tsne!')
    print(f'* embds shape: {embds.shape}')
    
    return embds


def prepare_largevis(embeddings, input_path):
    # Save the NumPy array to a text file
    # Prepare the shape information
    shape_info = f"{embeddings.shape[0]} {embeddings.shape[1]}\n"

    # Open the file in write mode and write the shape information as the first line
    with open(input_path, 'w') as f:
        f.write(shape_info)

    # Append the NumPy array to the text file
    with open(input_path, 'ab') as f:
        np.savetxt(f, embeddings, fmt='%f', delimiter='\t')

    print('* created %s!' % input_path)

def load_largevis(output_path):
    # Read the first line and split it to get the shape information
    with open(output_path, 'r') as f:
        shape_info = f.readline().strip().split(' ')
        rows, cols = int(shape_info[0]), int(shape_info[1])

    # Initialize an empty NumPy array with the extracted shape
    embds = np.empty((rows, cols))

    # Read the remaining lines and populate the NumPy array
    with open(output_path, 'r') as f:
        # Skip the first line (shape information)
        next(f)

        # Read the remaining lines and populate the NumPy array
        for i, line in enumerate(f):
            embds[i, :] = np.fromstring(line.strip(), sep='\t')

    print('* loaded 2d embds!')
    return embds

@timing_decorator
def calc_largevis(embeddings):
    # get number of threads
    available_cores = multiprocessing.cpu_count()
    print('* found %s CPU cores' % available_cores)
    n_thread = max(4, available_cores - 2)
    print("* set n_thread=%s" % n_thread)

    # get temp file path
    with tempfile.NamedTemporaryFile(delete=False) as f:
        input_path = f.name
        print('* made tmp file for input %s' % input_path)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        output_path = f.name
        print('* made tmp file for output %s' % output_path)

    # prepare input
    prepare_largevis(embeddings, input_path)
    
    # largevis it!
    print('* start LargeVis!')
    command = f'LargeVis -input {input_path} -output {output_path} -threads {n_thread}'
    subprocess.run(command, shell=True)
    print('* done LargeVis!')
    
    # load output
    embds = load_largevis(output_path)
    print(f'* embds shape: {embds.shape}')

    return embds


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
    parser = argparse.ArgumentParser(description="dimension reduction")
    parser.add_argument("--input_file_path", "-i", type=str, required=True, help="input file path containing the embeddings")
    parser.add_argument("--dimreduct_method", "-dr", choices={"tsne","largevis"}, default="largevis", help="dimension reduction method, i.e. tsne or largevis")
    parser.add_argument("--output_file_path", "-o", type=str, required=True, help="output file path containing the embds")
    
    args = parser.parse_args()
    input_file_path = args.input_file_path
    dimreduct_method = args.dimreduct_method
    output_file_path = args.output_file_path

    # embeddings load
    embeddings = pd.read_csv(input_file_path, sep='\t').values
    print(f"* loaded {embeddings.shape} from {input_file_path}")

    # dimension reduction
    if dimreduct_method == "tsne":
        random.seed(42)
        embds = calc_tsne(embeddings)
    elif dimreduct_method == "largevis":
        random.seed(42)
        embds = calc_largevis(embeddings)

    # embds save
    array_save(embds, output_file_path)


if __name__ == "__main__":
    main()