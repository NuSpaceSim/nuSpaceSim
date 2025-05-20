import uproot
import awkward as ak
import os
import glob
import re

def extract_numbers(filename):
    """
    Extract the numbers after 'n' and 'lgE' from a filename like 'nss_n8732_lgE19.root'.
    Returns a tuple (n_number, lgE_number).
    """
    n_match = re.search(r'n(\d+)', filename)
    lgE_match = re.search(r'lgE(\d+)', filename)
    
    if not n_match or not lgE_match:
        raise ValueError(f"Filename {filename} does not match expected format 'nss_nNUMBER_lgENUMBER.root'")
    
    n_number = int(n_match.group(1))
    lgE_number = int(lgE_match.group(1))
    return n_number, lgE_number

def combine_root_files(file_paths):
    # Check if there are files to process
    if not file_paths:
        print("No .root files found in the directory.")
        return

    # Open all input files
    files = [uproot.open(file_path) for file_path in file_paths]

    # Define the TTrees to combine
    tree_names = ["Shower", "Header"]

    # Check if both TTrees exist in all files
    for tree_name in tree_names:
        for file_path, file in zip(file_paths, files):
            if tree_name not in file:
                print(f"Error: TTree '{tree_name}' not found in {file_path}.")
                print(f"Available keys in {file_path}: {file.keys()}")
                for f in files:
                    f.close()
                return

    # Extract numbers from all filenames
    try:
        n_numbers, lgE_numbers = [], []
        for file_path in file_paths:
            n, lgE = extract_numbers(file_path)
            n_numbers.append(n)
            lgE_numbers.append(lgE)
    except ValueError as e:
        print(e)
        for f in files:
            f.close()
        return

    # Sum the 'n' numbers
    n_sum = sum(n_numbers)

    # Check if all lgE values are the same
    if len(set(lgE_numbers)) > 1:
        print(f"Warning: lgE values are not consistent across files: {lgE_numbers}")
        proceed = input("Do you want to proceed? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Aborting operation.")
            for f in files:
                f.close()
            return
        lgE_value = input(f"Please enter the lgE value to use in the output filename (e.g., 19 for lgE19): ").strip()
        try:
            lgE_value = int(lgE_value)
        except ValueError:
            print("Invalid lgE value entered. Aborting.")
            for f in files:
                f.close()
            return
    else:
        lgE_value = lgE_numbers[0]  # Use the common lgE value

    # Construct the output filename
    output_path = f"nss_n{n_sum}_lgE{lgE_value}.root"

    # Create the output file
    with uproot.recreate(output_path) as output_file:
        # Process each TTree
        for tree_name in tree_names:
            # Get arrays from all files for this tree
            all_arrays = [file[tree_name].arrays() for file in files]

            # Check if branch structures match across all files
            reference_fields = set(all_arrays[0].fields)
            for i, arrays in enumerate(all_arrays[1:], 1):
                if set(arrays.fields) != reference_fields:
                    print(f"Error: Branch structure mismatch in TTree '{tree_name}' for {file_paths[i]}.")
                    print(f"Branches in {file_paths[0]}: {reference_fields}")
                    print(f"Branches in {file_paths[i]}: {arrays.fields}")
                    for f in files:
                        f.close()
                    return

            # Combine arrays using awkward's concatenate
            combined_arrays = {}
            for branch in reference_fields:
                combined_arrays[branch] = ak.concatenate([arrays[branch] for arrays in all_arrays])

            # Write the combined arrays to the output file
            output_file[tree_name] = combined_arrays

    # Close all input files
    for f in files:
        f.close()
    print(f"Successfully combined {len(file_paths)} files into {output_path}")

if __name__ == "__main__":
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Find all .root files in the script's directory
    file_paths = glob.glob(os.path.join(script_dir, "*.root"))
    
    # Call the combining function with all .root files
    combine_root_files(file_paths)