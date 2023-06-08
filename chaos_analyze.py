import pandas as pd
import random
from collections import Counter
import csv

# Read the data from the CSV file
data = pd.read_csv('deuplasena6em50.csv', delimiter=';', encoding='latin-1')

# Remove blank rows and headers
data = data.dropna()
data = data.iloc[1:]

# Convert the data to a set of tuples
existing_sequences = {tuple(row) for row in data.values}


sequences = [list(row) for row in data.values]

# Number of combinations to generate
num_combinations = 1000000

# Compute the frequencies of observed sequences
sequence_freqs = Counter(tuple(seq) for seq in sequences)

# Open the CSV file for writing
with open('combidupla2.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=';')

    # Write the header row
    writer.writerow(['Number1', 'Number2', 'Number3', 'Number4', 'Number5'])

    # Generate and write the combinations
    for _ in range(num_combinations):
        next_sequence = random.choices(range(1, 50), k=6)
        sorted_sequence = sorted(next_sequence)
        writer.writerow(sorted_sequence)

print("Combinations saved to combinations.csv.")