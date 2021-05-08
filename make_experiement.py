import os
import argparse
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArgParser")
    parser.add_argument('exp_dir', type=str, help='The directory to create')
    parser.add_argument('--overwrite', action='store_const', const=True, default=False, help='will obliterate specified directory if it exists')
    args = parser.parse_args()

    p = os.path.join('experiments', args.exp_dir)

    if os.path.exists(p):
        if args.overwrite:
            shutil.rmtree(p)
        else:
            print(f"directory `{args.exp_dir}` already exists. Use `--overwrite' to replace that directory.")
            exit()


    os.makedirs(p, exist_ok=True)
    shutil.copyfile("default_config.json", os.path.join('experiments', args.exp_dir, "train_config.json"))
    shutil.copyfile("default_config.json", os.path.join('experiments', args.exp_dir, "eval_config.json"))
