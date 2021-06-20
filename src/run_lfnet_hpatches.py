import os 
import sys
import glob
import argparse
import subprocess


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", 
        help="directory of data",
    )
    args = parser.parse_args()
    return args 

def _main():
    args = _parse_arguments()
    
    seq_dirs = glob.glob(os.path.join(args.data_dir, "*"))
    seq_dirs = [os.path.abspath(path) for path in seq_dirs]

    print(f"Number of sequences: {len(seq_dirs)}")
    
    for seq_dir in seq_dirs:
        seq_name = os.path.basename(seq_dir)
        print(f"Extracting {seq_name}")
        subprocess.run(
            " ".join([
                "python3 /tmp2/b06902059/lf-net-release/run_lfnet.py",
                f"--in_dir {seq_dir}",
                f"--out_dir {seq_dir}",
                f"--full_output False"
            ]),
            shell=True
        )


if __name__ == "__main__":
    _main()    