#!/usr/bin/env python3
import pandas as pd
import os
import sys

if len(sys.argv) != 4:
    print("Usage: python script.py input.csv output_dir output_filename")
    sys.exit(1)

input_csv = sys.argv[1]
output_dir = sys.argv[2]
output_filename = sys.argv[3]

print(f"Input CSV: {input_csv}")
print(f"Output directory: {output_dir}")
print(f"Output filename: {output_filename}")

# Read the CSV file
print("Reading CSV file...")
data = pd.read_csv(input_csv)

# Select only 'path' and 'num_frames' columns
print("Selecting columns...")
selected_data = data[['path', 'num_frames']]

# Create output directory
print(f"Creating directory: {output_dir}")
os.makedirs(output_dir, exist_ok=True)

# Save as TSV
output_path = os.path.join(output_dir, output_filename)
print(f"Saving to: {output_path}")
selected_data.to_csv(output_path, sep='\t', index=False)

# Verify file was created
if os.path.exists(output_path):
    print(f"Successfully created: {output_path}")
    print(f"File size: {os.path.getsize(output_path)} bytes")
else:
    print("Error: File was not created")