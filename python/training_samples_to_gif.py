from os import listdir, system
import re


def main():
    samples = '1506352644'
    crop = 200
    samples_dir = './samples/{}'.format(samples)
    images_per_file = (6, 6)
    image_files = [f for f in listdir(samples_dir)]
    # [print(int(re.search('train_(\d+)_\d+.png', name).group(1))) for name in image_files]
    image_files.sort(key=lambda name: int(re.search('train_(\d+)_\d+.png', name).group(1)))
    image_files = image_files[0:crop]

    # image_files = ['{}/{}'.format(samples_dir, f) for f in image_files]
    image_magic_binary = '.\..\..\ImageMagick\convert.exe'
    command = '{} -loop 0 -delay 10 {} ./../../gifs/{}.gif'.format(image_magic_binary, ' '.join(image_files), samples)
    system("cd {} && {}".format(samples_dir, command))
    print('gif generated')


if __name__ == '__main__':
    main()