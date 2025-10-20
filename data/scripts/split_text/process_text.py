import argparse
import os
from pathlib import Path


def extract_text_before_hash(line: str) -> str:
    """
    Return the substring of the line before the first '#' character.
    If '#' is not present, return the line as-is.
    Trailing newlines are removed; callers should re-add newlines when writing.
    """
    hash_index = line.find('#')
    content = line if hash_index == -1 else line[:hash_index]
    # Remove trailing newlines and carriage returns to normalize; caller will add '\n' back
    return content.rstrip('\r\n')


def process_file(input_file: Path, output_file: Path) -> None:
    """
    Read an input text file line-by-line, write only the portion before '#' per line
    to the output file. Creates parent directories for the output file if needed.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with input_file.open('r', encoding='utf-8') as src, output_file.open('w', encoding='utf-8') as dst:
        for line in src:
            cleaned = extract_text_before_hash(line)
            dst.write(cleaned + '\n')


def process_directory(input_dir: Path, output_dir: Path) -> int:
    """
    Process all .txt files directly under input_dir and write to output_dir with the same filenames.
    Returns the number of files processed.
    """
    count = 0
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found or not a directory: {input_dir}")

    for entry in sorted(input_dir.iterdir()):
        if entry.is_file() and entry.suffix.lower() == '.txt':
            output_file = output_dir / entry.name
            process_file(entry, output_file)
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text before '#' for each line in all .txt files of a directory."
    )
    default_input = Path('data/hml3d/texts')
    default_output = Path('data/hml3d/texts_processed')
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=default_input,
        help=f"Input directory containing .txt files (default: {default_input})",
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=default_output,
        help=f"Output directory to write processed files (default: {default_output})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    processed_count = process_directory(input_dir, output_dir)
    print(f"Processed {processed_count} files from '{input_dir}' to '{output_dir}'.")


if __name__ == '__main__':
    main()


