#!/usr/bin/env python3
import argparse
import yaml
import glob


def convert_file(fname):
    with open(fname) as f:
        data = yaml.load(f)
        for cat, items in data.items():
            srcname = fname.strip('.yaml').replace('clean', 'raw')
            for item in items:
                x1, y1 = item['x'], item['y']
                x2, y2 = x1 + item['width'], y1 + item['height']
                print(f'{srcname},{x1},{y1},{x2},{y2},{item["class"]}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    return parser.parse_args()




def main(args):
    for fname in glob.glob(f'{args.directory}/*.yaml'):
        convert_file(fname)


if __name__ == '__main__':
    main(parse_args())
