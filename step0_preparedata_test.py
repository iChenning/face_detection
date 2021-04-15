import argparse
import os


def run(args):
    if not os.path.exists(os.path.split(args.list_dir)[0]):
        os.makedirs(os.path.split(args.list_dir)[0])
    f_ = open(args.list_dir, "w")

    dataset_dir = args.dataset_dir
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            file_dir = os.path.join(root, file)
            if file_dir.endswith('.jpg') or file_dir.endswith('.JPG') or \
                file_dir.endswith('.bmp') or file_dir.endswith('.png'):
                f_.writelines(file_dir + "\n")
    f_.close()


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset_dir', type=str, default=r'C:\Users\amlogic\datasets\san_1920',
                       help='the path of dataset waiting to a list')
    parse.add_argument('--list_dir', type=str, default='./data_list/test_san-1920_list.txt')

    args = parse.parse_args()
    run(args)