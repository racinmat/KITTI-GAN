import matplotlib

# http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server

matplotlib.use('Agg')

import numpy as np
import pickle
from PIL import Image


if __name__ == '__main__':
    data_dir = 'data/extracted'
    sizes_x = np.empty((1, 0))
    sizes_y = np.empty((1, 0))
    drives = [
        'drive_0009_sync',
        'drive_0015_sync',
        'drive_0023_sync',
        'drive_0032_sync',
    ]

    input_prefix = 'tracklets_points_grayscale_bg_white_'
    output_prefix = 'tracklets_points_normalized_'

    new_size = (32, 32)
    # new_size = (64, 64)

    for drive in drives:
        filename = data_dir + '/' + input_prefix + drive + '.data'
        print("processing: " + filename)
        file = open(filename, 'rb')
        data = pickle.load(file)
        file.close()
        for i, pair in enumerate(data):
            img = Image.fromarray(pair['y'])
            ratio = min(new_size[0] / img.size[0], new_size[1] / img.size[1])

            # resize image
            # img.save('temp_orig.png')
            new_img = img.resize((round(img.size[0] * ratio), round(img.size[1] * ratio)))
            # new_img.save('temp_resized.png')

            # fill missing places with white
            white = {'L': 255, 'RGB': (255, 255, 255)}
            bg = Image.new(mode=img.mode, size=new_size, color=white[img.mode])
            bg.paste(new_img, (0, 0, new_img.size[0], new_img.size[1]))  # Not centered, top-left corner
            # bg.save('temp_resized_padded.png')

            pair['y'] = np.array(bg)
            pair['x'].append(new_img.size[0])
            pair['x'].append(new_img.size[1])

            pair['y'] = 255 - pair['y']     # inverting black and white

        filename = data_dir + '/' + output_prefix + drive + '_' + str(new_size[0]) + '_' + str(new_size[1]) + '.data'
        file = open(filename, 'wb')
        pickle.dump(data, file)
        print("data written to file: " + filename)
        file.close()
