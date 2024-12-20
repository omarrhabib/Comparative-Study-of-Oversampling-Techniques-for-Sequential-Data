import pandas as pd
from multiprocessing import Pool, cpu_count
import os

# Function to process a chunk of rows


def process_chunk(chunk):
    iio_chunk = []
    for index, row in chunk.iterrows():
        q1_input = ""
        q2_input = str(int(row["q1"])) + " "
        q3_input = str(int(row["q1"])) + " " + str(int(row["q2"])) + " "
        q4_input = str(int(row["q1"])) + " " + \
            str(int(row["q2"])) + " " + str(int(row["q3"])) + " "

        iio_chunk.append([row["class"], q1_input, str(int(row["q1"]))])
        iio_chunk.append([row["class"], q2_input, str(int(row["q2"]))])
        iio_chunk.append([row["class"], q3_input, str(int(row["q3"]))])
        iio_chunk.append([row["class"], q4_input, str(int(row["q4"]))])

    return iio_chunk


def parallel_processing(df, num_cores):
    chunk_size = len(df) // num_cores
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    with Pool(num_cores) as pool:
        results = pool.map(process_chunk, chunks)

    # Flatten the list of lists into a single list
    flattened_results = [item for sublist in results for item in sublist]
    return flattened_results


if __name__ == "__main__":
    df = pd.read_csv("df.csv")
    num_cores = cpu_count()

    print(f"Using {num_cores} cores...")
    results = parallel_processing(df, num_cores)

    # Convert the results to a DataFrame
    iio = pd.DataFrame(results, columns=["instruction", "input", "output"])

    # Save to JSON
    iio.to_json('iio_train.json', orient='records', lines=True)

    print("DONE!!")
