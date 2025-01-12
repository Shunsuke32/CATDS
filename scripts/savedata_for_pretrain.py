import pandas as pd
import os
import shutil
import argparse
from pathlib import Path

def extract_paths_from_data(data_str):
    """Extract file paths from the string representation of DataFrames"""
    try:
        # Split the data into chunks (each chunk contains one DataFrame)
        chunks = data_str.split('","')
        
        # Process each chunk
        all_paths = []
        for chunk in chunks:
            # Clean up the chunk
            chunk = chunk.strip('"\n').strip()
            if not chunk:
                continue
                
            # Find lines containing path and num_frames
            lines = chunk.split('\n')
            for line in lines:
                if '.wav' in line:
                    # Extract the wav file path
                    parts = line.split()
                    # Find the part that ends with .wav
                    for part in parts:
                        if part.endswith('.wav'):
                            all_paths.append(part.strip())
                            break
        
        return all_paths
    except Exception as e:
        print(f"Error parsing data: {str(e)}")
        return []

def copy_audio_files(file_paths, source_dir, target_dir):
    """Copy audio files from source directory to target directory"""
    # Create target directory if it doesn't exist
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    # Keep track of statistics
    success_count = 0
    error_count = 0
    errors = []
    
    total_files = len(file_paths)
    
    for file_path in file_paths:
        source_file = os.path.join(source_dir, file_path)
        target_file = os.path.join(target_dir, file_path)
        
        try:
            # Create target subdirectories if they don't exist
            Path(os.path.dirname(target_file)).mkdir(parents=True, exist_ok=True)
            
            # Copy the file
            shutil.copy2(source_file, target_file)
            success_count += 1
            
            # Print progress
            if success_count % 50 == 0:
                progress = (success_count / total_files) * 100
                print(f"Progress: {progress:.1f}% ({success_count}/{total_files} files copied)...")
                
        except Exception as e:
            error_count += 1
            errors.append(f"Error copying {file_path}: {str(e)}")
    
    return success_count, error_count, errors

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Copy audio files based on input data file containing file paths.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python savedata_for_pretrain.py --input data.csv --source /path/to/source --target /path/to/target
        """
    )
    
    # Add arguments
    parser.add_argument('--input', '-i', required=True,
                      help='Path to the input file containing file paths')
    parser.add_argument('--source', '-s',
                      default="/workspace/data/IndicSUPERB/kb_data_clean_m4a/hindi/train/audio",
                      help='Source directory containing the audio files')
    parser.add_argument('--target', '-t',
                      default="/workspace/data/IndicSUPERB/hindi/train/full",
                      help='Target directory where files will be copied')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        return
    
    # Check if source directory exists
    if not os.path.exists(args.source):
        print(f"Error: Source directory '{args.source}' does not exist.")
        return
    
    # Get the string data from your DataFrame
    try:
        with open(args.input, 'r') as f:
            data_str = f.read()
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        return
    
    # Extract file paths
    file_paths = extract_paths_from_data(data_str)
    
    # Remove any whitespace from file paths
    file_paths = [path.strip() for path in file_paths if path.strip()]
    
    print(f"Found {len(file_paths)} files to copy")
    print(f"Source directory: {args.source}")
    print(f"Target directory: {args.target}")
    print("\nStarting copy operation...")
    
    if not file_paths:
        print("No files to copy. Please check the input data format.")
        return
    
    # Copy the files
    success_count, error_count, errors = copy_audio_files(file_paths, args.source, args.target)
    
    # Print summary
    print("\nOperation completed!")
    print(f"Successfully copied: {success_count} files")
    print(f"Errors encountered: {error_count} files")
    
    if errors:
        print("\nErrors:")
        for error in errors:
            print(error)
            
    if success_count == len(file_paths):
        print("\nAll files were copied successfully!")
    else:
        print(f"\nWarning: {len(file_paths) - success_count} files were not copied.")

if __name__ == "__main__":
    main()