#!/usr/bin/env python3
"""
Split the enzyme CSV into 3 batches for processing
"""

import pandas as pd
import os

# Load the original CSV
input_file = "enzyme_mutations_clean.csv"
output_dir = "data"

print(f"📂 Loading {input_file}...")
df = pd.read_csv(input_file)
total_rows = len(df)
print(f"📊 Total rows: {total_rows}")

# Calculate batch sizes
batch_size = 700  # Max 700 per batch (to stay under 1000 daily limit with margin)
num_batches = 3

# Split into batches
batches = []
for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, total_rows)

    if start_idx < total_rows:
        batch = df.iloc[start_idx:end_idx]
        batches.append(batch)
        print(f"📦 Batch {i+1}: rows {start_idx}-{end_idx-1} ({len(batch)} rows)")

# Save each batch
for i, batch in enumerate(batches, 1):
    output_file = os.path.join(output_dir, f"enzyme_mutations_batch_{i}.csv")
    batch.to_csv(output_file, index=False)
    print(f"💾 Saved {output_file}")

# Create a summary
print("\n" + "="*50)
print("📈 Summary:")
print(f"  Total rows: {total_rows}")
print(f"  Number of batches: {len(batches)}")
for i, batch in enumerate(batches, 1):
    print(f"  Batch {i}: {len(batch)} rows")

print("\n✅ CSV splitting complete!")
print(f"📁 Files saved in: {output_dir}/")