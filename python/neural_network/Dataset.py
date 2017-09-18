import numpy as np


class DataSet(object):

    num_examples = 0
    images = None
    labels = None
    epochs_completed = 0
    index_in_epoch = 0

    def __init__(self, data):
        # Shuffle the data
        """Construct a DataSet. one_hot arg is used only if fake_data is true."""

        images = np.array([t['y'] for t in data])
        labels = np.array([t['x'] for t in data])  # removed angle for this first iteration

        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)

        # because it awaits image to be 3D array (w x h x channels)
        if len(images.shape) < 4:
            images = np.expand_dims(images, axis=3)
        images = np.multiply(images, 1.0 / 255.0)

        self.num_examples = len(data)
        perm = np.arange(self.num_examples)
        np.random.shuffle(perm)

        self.images = images[perm]
        self.labels = labels[perm]
        self.epochs_completed = 0
        self.index_in_epoch = 0

    def split(self, ratios):
        if sum(ratios) is not 1:
            raise Exception("Wrong ratios sum")

        del ratios[-1]  # remove lasst ratio
        splitters = []
        for ratio in ratios:
            splitters.append(int(ratio * self.num_examples))

        data = np.array([{'x': label, 'y': image} for image, label in zip(self.images, self.labels)])

        data_sets = np.split(data, splitters)
        return [DataSet(dataset) for dataset in data_sets]

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.labels = self.labels[perm]
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self.index_in_epoch
        return self.images[start:end], self.labels[start:end]

    def num_batches(self, batch_size):
        return np.floor(self.num_examples / batch_size)

    def get_image_size(self):
        return self.images[0].shape

    def get_labels_dim(self):
        return len(self.labels[0])
