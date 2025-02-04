import numpy as np
import csv


# Function to generate a random sequence similar to the given format
def generate_sequence():
    num = 500  # Number of samples to generate

    # Open file safely with 'with open' (auto-closes)
    with open("s_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        
        # Write the header
        writer.writerow(["Sample_ID", "initial_amount", "sample_concentration", "spore_count"])
        
        # Generate random values
        for line in range(num):
            sample_id = line  # Unique ID for each sample
            initial_amount = np.random.randint(98, 102)
            sample_concentration = np.random.randint(1, 5)  # (1-5) 

            # Conditional fluorescence intensity and bacterial reduction
            if sample_concentration > 2:
                spore_count = np.random.randint(0, 20)  # (0-20)
            else:
                spore_count = np.random.randint(0, 200)  # (0-200)

            # Write row to CSV
            writer.writerow([sample_id, initial_amount, sample_concentration, spore_count])

# Call function to generate the file
generate_sequence()
