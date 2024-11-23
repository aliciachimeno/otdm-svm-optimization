import numpy as np

def clean_data(input_file):
    """
    Cleans the input file by removing or replacing any non-numeric values.
    Returns a list of cleaned data rows.
    """
    cleaned_lines = []
    with open(input_file, 'r') as file:
        for line in file:
            # Replace non-numeric values or symbols (like '1.0*' to '1.0')
            line = line.strip()
            # Remove any unwanted characters using a regular expression (if needed)
            cleaned_line = ''.join(c if c.isdigit() or c in '.- \t' else '' for c in line)
            cleaned_lines.append(cleaned_line)
    
    return cleaned_lines

def read_input_file(input_file):
    """
    Reads the cleaned .txt file and returns the feature matrix and targets.
    """
    # Clean the data first
    cleaned_data = clean_data(input_file)
    
    # Now we read the cleaned data into a numpy array
    data = np.array([list(map(float, line.split())) for line in cleaned_data if line.strip()])
    
    features = data[:, :-1]  # All columns except the last one
    targets = data[:, -1]  # The last column as the target
    return features, targets

def write_dat_file(output_file, X_train, y_train, X_test, y_test, m, n, nu):
    """
    Write the data in the specified .dat file format.
    """
    with open(output_file, 'w') as f:
        # Writing header and dimensions
        f.write("# Train and test dataset\n")
        f.write("# 50/50\n")
        f.write("# Seed 66407\n\n")
        f.write(f"# Dimensions(m,n)\nparam m := {m};\nparam n := {n};\n\n")
        f.write(f"param nu := {nu};\n\n")

        # Writing y_train
        f.write("# y_train\n")
        f.write("param y_train :=\n")
        for i in range(len(y_train)):
            f.write(f"{i + 1}\t{y_train[i]}\n")
        f.write(";\n\n")

        # Writing A_train
        f.write("# A_train\n")
        f.write(f"param A_train : 1 2 3 4 :=\n")
        for i in range(len(X_train)):
            f.write(f"{i + 1}\t")
            f.write("\t".join(map(str, X_train[i])))
            f.write("\n")
        f.write(";\n\n")

        # Writing y_test
        f.write("# y_test\n")
        f.write("param y_test :=\n")
        for i in range(len(y_test)):
            f.write(f"{i + 1}\t{y_test[i]}\n")
        f.write(";\n\n")

        # Writing A_test
        f.write("# A_test\n")
        f.write(f"param A_test : 1 2 3 4 :=\n")
        for i in range(len(X_test)):
            f.write(f"{i + 1}\t")
            f.write("\t".join(map(str, X_test[i])))
            f.write("\n")
        f.write(";\n")

def convert_to_dat(input_file, output_file, test_size=0.5, nu=0.9):
    """
    Reads the .txt file, splits the data into train/test, and writes it to a .dat file.
    """
    # Read the input file
    features, targets = read_input_file(input_file)

    # Split data into train and test
    m = int(len(features) * (1 - test_size))  # number of training samples
    n = features.shape[1]  # number of features

    # Train/Test split (simple slicing, here we use the first 50% for training and the rest for testing)
    X_train = features[:m]
    y_train = targets[:m]
    X_test = features[m:]
    y_test = targets[m:]

    # Write the formatted data to the .dat file
    write_dat_file(output_file, X_train, y_train, X_test, y_test, m, n, nu)
    print(f"Data has been written to {output_file}")

# Example Usage
input_file = "raw/data1000.txt"  # Input dataset file (the original data)
output_file = "ampl_format/data1000_converted.dat"  # Output AMPL-compatible .dat file
convert_to_dat(input_file, output_file)
