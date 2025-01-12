import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def convert_m4a_to_wav(input_file: Path) -> None:
    """
    Convert a single M4A file to 16 kHz mono WAV format using ffmpeg,
    overwriting any existing WAV files
    
    Args:
        input_file (Path): Path to input M4A file
    """
    output_file = input_file.with_suffix('.wav')
    
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', str(input_file),
        '-acodec', 'pcm_s16le',  # 16-bit PCM
        '-ac', '1',              # mono
        '-ar', '16000',          # 16 kHz
        '-y',                    # Overwrite output file if exists
        str(output_file)
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_file.name}: {e}")

def main():
    # Input directory path
    input_dir = Path('/work/data/IndicSUPERB/kb_data_clean_m4a/punjabi/train/audio')
    
    # Get all M4A files
    m4a_files = list(input_dir.glob('*.m4a'))
    
    if not m4a_files:
        print("No M4A files found in the directory.")
        return
        
    print(f"Found {len(m4a_files)} M4A files. Starting conversion...")
    
    # Convert files using thread pool for parallel processing
    with ThreadPoolExecutor() as executor:
        list(tqdm(
            executor.map(convert_m4a_to_wav, m4a_files),
            total=len(m4a_files),
            desc="Converting files"
        ))
    
    print("Conversion completed!")

if __name__ == '__main__':
    main()