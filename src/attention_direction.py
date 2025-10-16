import torch
import os
import argparse
from tqdm import tqdm

def get_age_group(age: int) -> str:
    """Categorize age into young/old groups"""
    if age <= 45:
        return "young"
    elif age >= 60:
        return "old"
    else:
        return "unknown"




def compute_group_average(group_files, output_path):
    """
    Compute averaged features for a group incrementally
    Args:
        group_files: List of file paths for the group
        output_path: Path to save averaged features
    """
    # Initialize accumulation dictionaries
    sum_dict = None
    count_dict = {}

    for file_path in tqdm(group_files, desc=f"Averaging {os.path.basename(output_path)}"):
        # Added check for file existence inside loop
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}, skipping.")
            continue
        try:
            features = torch.load(file_path, map_location='cpu', weights_only=True) # Load to CPU first
        except Exception as e:
            print(f"Error loading {file_path}: {e}, skipping.")
            continue

        if sum_dict is None:  # Initialize on first iteration based on first valid file
            sum_dict = {k: torch.zeros_like(v, dtype=torch.bfloat16, device='cpu') for k, v in features.items()}
            count_dict = {k: 0 for k in features.keys()}

        # Accumulate sums and counts (ensure keys match initialization)
        for k, v in features.items():
            if k in sum_dict:
                sum_dict[k] += v.to(torch.bfloat16) # Keep on CPU for accumulation
                count_dict[k] += 1
            else:
                print(f"Warning: Key {k} from {file_path} not in initial feature set, skipping.")


        del features
        torch.cuda.empty_cache()
    
    # Compute averages and convert back to original dtype
    # Handle cases where a group might have had no valid files
    if sum_dict is None:
        print(f"Error: No valid feature files found for {output_path}. Cannot compute average.")
        return False

    avg_features = {}
    for k in sum_dict:
         if count_dict[k] > 0:
             avg_features[k] = (sum_dict[k] / count_dict[k]).to(torch.bfloat16)
         else:
             print(f"Warning: No data accumulated for key {k} in {output_path}.")

    if not avg_features:
        print(f"Error: No features could be averaged for {output_path}.")
        return False


    # Save averaged features
    try:
        torch.save(avg_features, output_path)
        print(f"Saved averaged features to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving {output_path}: {e}")
        return False


def process_attention_type(attention_type, base_path, output_dir_averages, output_dir_direction):
    """Groups files, computes averages, calculates, and saves editing direction for a given attention type."""
    print(f"--- Processing Attention Type: {attention_type} ---")

    # 1. Group files specific to this attention type
    age_groups = {"young": [], "old": []}
    print(f"Scanning {base_path} for features_{attention_type}.pth...")
    found_files = False
    if not os.path.isdir(base_path):
        print(f"Error: Base path {base_path} not found or is not a directory.")
        return

    for age_dir_name in os.listdir(base_path):
        try:
            age_str = age_dir_name.split('_')[0]
            if not age_str.isdigit():
                continue
            age = int(age_str)
        except (IndexError, ValueError):
            continue

        group = get_age_group(age)
        if group == "unknown":
            continue

        age_dir_path = os.path.join(base_path, age_dir_name)
        if not os.path.isdir(age_dir_path):
            continue

        file_path = os.path.join(age_dir_path, f"features_{attention_type}.pth")
        if os.path.exists(file_path):
            age_groups[group].append(file_path)
            found_files = True

    if not found_files:
        print(f"Warning: No feature files found for attention type '{attention_type}' in {base_path}.")

    # 2. Compute group averages
    os.makedirs(output_dir_averages, exist_ok=True)
    average_computed = {"young": False, "old": False}
    for group_name, file_paths in age_groups.items():
        if not file_paths:
            print(f"Warning: No files found for group '{group_name}' and attention type '{attention_type}'. Skipping averaging.")
            continue
        output_path = os.path.join(output_dir_averages, f"{group_name}_average_{attention_type}.pth")
        average_computed[group_name] = compute_group_average(file_paths, output_path)

    # 3. Calculate editing direction (only if both averages were successfully computed)
    if not average_computed["young"] or not average_computed["old"]:
         print(f"Warning: Could not compute both young and old averages for attention type '{attention_type}'. Skipping editing direction calculation.")
         return

    young_avg_path = os.path.join(output_dir_averages, f"young_average_{attention_type}.pth")
    old_avg_path = os.path.join(output_dir_averages, f"old_average_{attention_type}.pth")

    # Double check files exist before loading
    if not os.path.exists(young_avg_path) or not os.path.exists(old_avg_path):
        print(f"Error: Average files missing despite reporting success for attention type '{attention_type}'. Cannot compute editing direction.")
        return

    try:
        avg_features_young = torch.load(young_avg_path, map_location='cpu')
        avg_features_old = torch.load(old_avg_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading average files for {attention_type}: {e}")
        return

    editing_direction = {k: avg_features_old[k].to(torch.bfloat16) - avg_features_young[k].to(torch.bfloat16)
                         for k in avg_features_old.keys() if k in avg_features_young} # Ensure key exists in both

    # Check if any keys were mismatched
    if len(editing_direction) != len(avg_features_old) or len(editing_direction) != len(avg_features_young):
        print(f"Warning: Key mismatch between young and old averages for {attention_type}. Direction calculated only for common keys.")


    # 4. Save editing direction components
    os.makedirs(output_dir_direction, exist_ok=True)
    print(f"Saving editing directions for {attention_type} to {output_dir_direction}...")
    saved_count = 0
    for k, v in editing_direction.items():
        try:
            # Save with attention type in the filename to avoid overwrites
            save_path = os.path.join(output_dir_direction, f"{k}.pth")
            torch.save(v, save_path)
            saved_count += 1
        except Exception as e:
            print(f"Error saving editing direction component {k}.pth: {e}")
    print(f"Saved {saved_count} editing direction components for {attention_type}.")


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute aging direction from extracted features')
    
    parser.add_argument('--base_path', type=str, required=True,
                        help='Path to directory containing extracted features (e.g., /path/to/features_young/person_kv)')
    parser.add_argument('--output_dir_averages', type=str, required=True,
                        help='Output directory for age group averages (e.g., /path/to/age_groups_kv)')
    parser.add_argument('--output_dir_direction', type=str, required=True,
                        help='Output directory for aging direction features (e.g., /path/to/editing_direction_kv)')
    
    args = parser.parse_args()
    
    # Loop through attention types
    for attn_type in ['K', 'V']:
        process_attention_type(attn_type, args.base_path, args.output_dir_averages, args.output_dir_direction)
    
    print("--- Processing Complete ---")
