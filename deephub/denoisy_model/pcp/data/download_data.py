import os
#import zipfile
import urllib
import tarfile
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    # naming / file handling
    parser.add_argument(
        '--task', type=str, default='denoising', help='task name for dataset')
    return parser.parse_args()

def download_dataset(source_url, target_dir, target_file):
    global downloaded
    downloaded = 0
    def show_progress(count, block_size, total_size):
        global downloaded
        downloaded += block_size
        print('downloading ... %d%%' % round(((downloaded*100.0) / total_size)))

    print('downloading ... ')
    urllib.urlretrieve(source_url, filename=target_file, reporthook=show_progress)
    print('downloading ... done')

    print('extracting ...')
    tar = tarfile.open(target_file, "r:gz")
    tar.extractall()
    tar.close()
    os.remove(target_file)
    print('extracting ... done')


if __name__ == '__main__':
    opt = parse_arguments()
    if opt.task == "denoising":
        source_url = 'http://geometry.cs.ucl.ac.uk/projects/2019/pointcleannet/data/pointCleanNetDataset.tar.gz'
        target_dir = os.path.dirname(os.path.abspath(__file__))
        target_file = os.path.join(target_dir, 'pointCleanNetDataset.tar.gz')
        download_dataset(source_url,  target_dir, target_file)
    elif opt.task == "outliers_removal":
        source_url = 'http://geometry.cs.ucl.ac.uk/projects/2019/pointcleannet/data/pointCleanNetOutliersTestSet.tar.gz'
        target_dir = os.path.dirname(os.path.abspath(__file__))
        target_file = os.path.join(target_dir, 'pointCleanNetOutliersTestSet.tar.gz')
        download_dataset(source_url,  target_dir, target_file)
        source_url = 'http://geometry.cs.ucl.ac.uk/projects/2019/pointcleannet/data/pointCleanNetOutliersTrainingSet.tar.gz'
        target_dir = os.path.dirname(os.path.abspath(__file__))
        target_file = os.path.join(target_dir, 'pointCleanNetOutliersTrainingSet.tar.gz')
        download_dataset(source_url,  target_dir, target_file)
    else:
        print('unknown dataset')
