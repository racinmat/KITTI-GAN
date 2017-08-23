import numpy as np


class DataSet(object):

    num_examples = 0
    images = None
    labels = None
    epochs_completed = 0
    index_in_epoch = 0

    def __init__(self, data):
        """Construct a DataSet. one_hot arg is used only if fake_data is true."""

        self.num_examples = len(data)

        images = np.array([t['y'] for t in data])
        labels = np.array([t['x'] for t in data])  # removed angle for this first iteration

        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        if images.shape[2] == 1:
            images = np.expand_dims(images, axis=3)       # because it awaits image to be 3D array (w x h x channels)
        images = np.multiply(images, 1.0 / 255.0)

        self.images = images
        self.labels = labels
        self.epochs_completed = 0
        self.index_in_epoch = 0

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
