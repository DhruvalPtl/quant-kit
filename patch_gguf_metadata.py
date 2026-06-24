import os
import sys
import argparse
import numpy as np
from gguf.gguf_reader import GGUFReader

def patch_gguf(file_path: str, dry_run: bool = False):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return False

    print(f"Opening GGUF file: {file_path}")
    mode = 'r' if dry_run else 'r+'
    
    try:
        reader = GGUFReader(file_path, mode=mode)
    except Exception as e:
        print(f"Failed to parse GGUF file: {e}")
        return False

    print("\n--- Current Metadata ---")
    arch_field = reader.fields.get('general.architecture')
    arch = None
    if arch_field:
        arch = arch_field.contents()
        print(f"Architecture: {arch}")
    
    block_count_key = f"{arch}.block_count" if arch else "qwen35moe.block_count"
    nextn_key = f"{arch}.nextn_predict_layers" if arch else "qwen35moe.nextn_predict_layers"

    block_field = reader.fields.get(block_count_key)
    nextn_field = reader.fields.get(nextn_key)

    if block_field:
        print(f"{block_count_key}: {block_field.contents()}")
    else:
        print(f"Warning: {block_count_key} not found")

    if nextn_field:
        print(f"{nextn_key}: {nextn_field.contents()}")
    else:
        print(f"Warning: {nextn_key} not found")

    if dry_run:
        print("\nDry-run completed. No changes made.")
        return True

    modified = False
    
    # Update block count from 41 to 40
    if block_field:
        current_val = block_field.contents()
        if current_val == 41:
            print(f"\nPatching {block_count_key} from 41 to 40...")
            block_field.parts[-1][0] = 40
            modified = True
            print(f"Verification: {block_field.contents()}")
        else:
            print(f"\n{block_count_key} is {current_val}, no need to patch (expected 41).")

    # Update nextn predict layers from 1 to 0
    if nextn_field:
        current_val = nextn_field.contents()
        if current_val == 1:
            print(f"Patching {nextn_key} from 1 to 0...")
            nextn_field.parts[-1][0] = 0
            modified = True
            print(f"Verification: {nextn_key}: {nextn_field.contents()}")
        else:
            print(f"{nextn_key} is {current_val}, no need to patch (expected 1).")

    if modified:
        # Flush memmap changes to disk
        reader.data.flush()
        print("\nSuccess! GGUF file patched successfully in-place.")
    else:
        print("\nNo modifications were required.")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch GGUF block count and MTP settings in-place")
    parser.add_argument("file", help="Path to the GGUF file")
    parser.add_argument("--dry-run", action="store_true", help="View current values without modifying")
    args = parser.parse_args()
    patch_gguf(args.file, args.dry_run)
