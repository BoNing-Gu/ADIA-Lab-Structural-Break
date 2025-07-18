import argparse
from train import train

def main():
    parser = argparse.ArgumentParser(description="Deep Learning Experiment for Structural Break Detection")
    parser.add_argument('--mode', type=str, default='train', choices=['train'],
                        help="Mode to run: 'train'")
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train()

if __name__ == '__main__':
    main() 