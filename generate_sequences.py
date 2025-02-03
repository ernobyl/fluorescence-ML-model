import numpy as np
import csv


# Function to generate a random sequence similar to the given format
def generate_sequence():
    num = 500  # Number of samples to generate

    # Open file safely with 'with open' (auto-closes)
    with open("fluorescence_experiment_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        
        # Write the header
        writer.writerow(["Sample_ID", "fluorescence_intensity", "treatment_time", "drug_concentration", "bacterial_reduction"])
        
        # Generate random values
        for line in range(num):
            sample_id = line  # Unique ID for each sample
            treatment_time = np.random.randint(2, 25)   # (2-25)
            drug_concentration = np.random.randint(3, 25)  # (3-25)

            # Conditional fluorescence intensity and bacterial reduction
            if treatment_time > 15 or drug_concentration > 15:
                fluorescence_intensity = np.random.randint(50, 125)  # (50-125)
                bacterial_reduction = np.random.randint(65, 100)  # (65-100)
            else:
                fluorescence_intensity = np.random.randint(100, 250)  # (100-250)
                bacterial_reduction = np.random.randint(20, 70)  # (20-70)

            # Write row to CSV
            writer.writerow([sample_id, fluorescence_intensity, treatment_time, drug_concentration, bacterial_reduction])

# Call function to generate the file
generate_sequence()
