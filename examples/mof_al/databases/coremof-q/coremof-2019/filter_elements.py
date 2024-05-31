import argparse
import os
import shutil


def transfer_files(source, target, filter_fn):
    if not os.path.exists(target):
        os.mkdir(target)

    for f in os.listdir(source):
        s_path = os.path.join(source, f)
        t_path = os.path.join(target, f)
        if os.path.isdir(s_path):
            transfer_files(t_path, t_path, filter_fn)
        else:
            if filter_fn(s_path):
                shutil.copy(s_path, t_path)


def main(source, target, elements):
    def filter_fn(source):
        with open(source) as inp:
            content = inp.readlines()
        for i in range(len(content)):
            line = content[i].strip()
            if len(line) > 0 and line[0].isupper():
                break
        content = ''.join(content[i:])
        content = content.replace('Biso', '')
        if any([e in content for e in elements]):
            return False
        return True

    transfer_files(source, target, filter_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('target')
    parser.add_argument('-e', '--elements', nargs='+')
    main(**vars(parser.parse_args()))
