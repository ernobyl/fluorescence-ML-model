import numpy as np

# Function to generate a random sequence similar to the given format
def generate_sequence():
    num = 500
    f = open("fluorescence_experiment_data.csv", "w")
    print("Sample_ID,fluorescence_intensity,treatment_time,drug_concentration,bacterial_reduction", file=f)
    # Random values within the expected ranges
    for line in range(num):
        col1 = line  # Similar to the first column (1-1000)
        col2 = np.random.randint(50, 250)  # Similar to the second column (50-250)
        col3 = np.random.randint(2, 25)   # Similar to the third column (2-25)
        col4 = np.random.randint(3, 25)   # Similar to the fourth column (3-25)
        col5 = np.random.randint(20, 100) # Similar to the fifth column (20-100)
    
        print(col1, ",", col2, ",", col3, ",", col4, ",", col5, file=f)
    
    f.close()

generate_sequence()