import uproot
import awkward as ak
import argparse
import re

def extract_numbers(filename):
    """
    Extract the numbers after 'n' and 'lgE' from a filename like 'nss_n8732_lgE19.root'.
    Returns a tuple (n_number, lgE_number).
    """
    # Match the pattern nNUMBER and lgE_NUMBER
    n_match = re.search(r'n(\d+)', filename)
    lgE_match = re.search(r'lgE(\d+)', filename)
    
    if not n_match or not lgE_match:
        raise ValueError(f"Filename {filename} does not match expected format 'nss_nNUMBER_lgENUMBER.root'")
    
    n_number = int(n_match.group(1))  # Number after 'n'
    lgE_number = int(lgE_match.group(1))  # Number after 'lgE'
    return n_number, lgE_number

def combine_root_files(file1_path, file2_path):
    # Open the two input files
    file1 = uproot.open(file1_path)
    file2 = uproot.open(file2_path)

    # Define the TTrees to combine
    tree_names = ["Shower", "Header"]

    # Check if both TTrees exist in both files
    for tree_name in tree_names:
        if tree_name not in file1 or tree_name not in file2:
            print(f"Error: TTree '{tree_name}' not found in one of the files.")
            print(f"Available keys in {file1_path}: {file1.keys()}")
            print(f"Available keys in {file2_path}: {file2.keys()}")
            file1.close()
            file2.close()
            return

    # Extract numbers from both filenames
    try:
        n1, lgE1 = extract_numbers(file1_path)
        n2, lgE2 = extract_numbers(file2_path)
    except ValueError as e:
        print(e)
        file1.close()
        file2.close()
        return

    # Sum the 'n' numbers
    n_sum = n1 + n2

    # Check if lgE values are the same
    if lgE1 != lgE2:
        print(f"Warning: lgE values are not the same ({lgE1} in {file1_path}, {lgE2} in {file2_path}).")
        proceed = input("Do you want to proceed? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Aborting operation.")
            file1.close()
            file2.close()
            return
        
        lgE_value = input(f"Please enter the lgE value to use in the output filename (e.g., 19 for lgE19): ").strip()
        try:
            lgE_value = int(lgE_value)
        except ValueError:
            print("Invalid lgE value entered. Aborting.")
            file1.close()
            file2.close()
            return
    else:
        lgE_value = lgE1  # Use the common lgE value

    # Construct the output filename
    output_path = f"nss_n{n_sum}_lgE{lgE_value}.root"

    # Create the output file
    with uproot.recreate(output_path) as output_file:
        # Process each TTree
        for tree_name in tree_names:
            # Access the TTrees
            tree1 = file1[tree_name]
            tree2 = file2[tree_name]

            # Get the arrays from both trees
            arrays1 = tree1.arrays()
            arrays2 = tree2.arrays()

            # Check if the branch structures match
            if set(arrays1.fields) != set(arrays2.fields):
                print(f"Error: Branch structure mismatch in TTree '{tree_name}'.")
                print(f"Branches in {file1_path}: {arrays1.fields}")
                print(f"Branches in {file2_path}: {arrays2.fields}")
                file1.close()
                file2.close()
                return

            # Combine the arrays using awkward's concatenate
            combined_arrays = {}
            for branch in arrays1.fields:
                combined_arrays[branch] = ak.concatenate([arrays1[branch], arrays2[branch]])

            # Write the combined arrays to the output file under the same TTree name
            output_file[tree_name] = combined_arrays

    # Close the input files
    file1.close()
    file2.close()
    print(f"Successfully combined {file1_path} and {file2_path} into {output_path}")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Combine two ROOT files into one")
    parser.add_argument("file1", help="Path to first ROOT file")
    parser.add_argument("file2", help="Path to second ROOT file")

    # Parse arguments
    args = parser.parse_args()

    # Call the combining function with provided arguments
    combine_root_files(args.file1, args.file2)