import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import keras

"""
Based on: https://github.com/emansim/unsupervised-videos
"""


class BouncingMNISTDataGenerator(keras.utils.Sequence):
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, seq_length=16, batch_size=32,
                 dataset_size=32000, image_size=64, digit_size=28, class_velocity=False, hide_digits=False,
                 num_digits=2, step_length=0.1, shuffle=True,
                 split='train', random_seed=42, ae=False, noise=False,
                 mnist_path='datasets/mnist/mnist.h5'):
        self.seq_length_ = seq_length
        self.batch_size_ = batch_size
        self.num_digits_ = num_digits
        self.class_velocity = class_velocity
        self.hide_digits = hide_digits
        self.step_length_ = step_length
        # The dataset is really infinite. This is just for validation.
        self.dataset_size_ = dataset_size
        self.digit_size_ = digit_size
        self.image_size_ = image_size
        self.random_seed_ = random_seed
        self.shuffle_ = shuffle
        self.ae_ = ae
        self.noise_ = noise
        self.n_classes = 10
        self.n_labels = num_digits

        try:
            f = h5py.File(mnist_path, 'r')
            self.data_ = f[split].value.reshape(-1, 28, 28, 1)
            self.labels_ = f[split + '_labels'].value
            f.close()
        except KeyError:
            print('Please set the correct path to MNIST dataset')
            sys.exit()

        self.on_epoch_end()

    def get_batch_size(self):
        return self.batch_size_

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.dataset_size_ // self.batch_size_)

    def get_dims(self):
        return self.image_size_

    def get_dataset_size(self):
        return self.dataset_size_

    def get_seq_length(self):
        return self.seq_length_

    def reset(self):
        pass

    def get_random_trajectory(self, batch_size, labels=None):
        length = self.seq_length_
        canvas_size = self.image_size_ - self.digit_size_

        # Initial position uniform random inside the box.
        y = np.random.rand(batch_size)
        x = np.random.rand(batch_size)

        # Choose a random velocity.
        theta = np.random.rand(batch_size) * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros((length, batch_size))
        start_x = np.zeros((length, batch_size))
        for i in range(length):
            # Take a step along velocity.
            if labels is not None:
                y += v_y * self.step_length_ * (labels+1)
                x += v_x * self.step_length_ * (labels+1)
            else:
                y += v_y * self.step_length_
                x += v_x * self.step_length_

            # Bounce off edges.
            for j in range(batch_size):
                if x[j] <= 0:
                    x[j] = 0
                    v_x[j] = -v_x[j]
                if x[j] >= 1.0:
                    x[j] = 1.0
                    v_x[j] = -v_x[j]
                if y[j] <= 0:
                    y[j] = 0
                    v_y[j] = -v_y[j]
                if y[j] >= 1.0:
                    y[j] = 1.0
                    v_y[j] = -v_y[j]
            start_y[i, :] = y
            start_x[i, :] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def overlap(self, a, b):
        """ Put b on top of a."""
        return np.maximum(a, b)

    def get_digits_idx(self):
        idx = np.zeros((self.batch_size_ * self.num_digits_),
                       dtype=np.int32)

        for ii in range(self.batch_size_ * self.num_digits_):
            idx[ii] = self.indices_[self.row_]
            self.row_ += 1
            if self.row_ == self.data_.shape[0]:
                self.row_ = 0
                if self.shuffle_:
                    np.random.shuffle(self.indices_)

        return idx

    def get_batch(self, ret_bbox=False, verbose=False):
        idx = self.get_digits_idx()
        if self.class_velocity is True:
            start_y, start_x = \
                self.get_random_trajectory(self.batch_size_ *
                                           self.num_digits_,
                                           labels=self.labels_[idx])
        else:
            start_y, start_x = \
                self.get_random_trajectory(self.batch_size_ *
                                           self.num_digits_)

        # minibatch data
        data = np.zeros((self.batch_size_, self.seq_length_,
                         self.image_size_, self.image_size_, 1),
                        dtype=np.float32)
        labels = np.zeros((self.batch_size_, self.seq_length_,
                           self.num_digits_, 5), dtype=np.float32)

        for j in range(self.batch_size_):
            for n in range(self.num_digits_):

                # get random digit from dataset
                if self.hide_digits is True:
                    digit_image = np.zeros(self.data_[0].shape,
                                           dtype=np.float32)
                    digit_image[7:-7, 7:-7, :] = \
                        np.ones((14, 14, 1),
                                dtype=np.float32)
                else:
                    digit_image = \
                        self.data_[idx[j*self.num_digits_+n], :, :, :]
                label = self.labels_[idx[j*self.num_digits_+n]]

                # generate video
                for i in range(self.seq_length_):
                    top = start_y[i, j * self.num_digits_ + n]
                    left = start_x[i, j * self.num_digits_ + n]
                    bottom = top + self.digit_size_
                    right = left + self.digit_size_
                    data[j, i, top:bottom, left:right, :] = self.overlap(
                        data[j, i, top:bottom, left:right, :], digit_image)

                    bbox = [top, bottom, left, right]
                    labels[j, i, n, 0] = label
                    labels[j, i, n, 1:5] = bbox

        if self.ae_ is True:
            labels = np.clip(np.sum(
                keras.utils.to_categorical(np.sort(labels[:, 0, :, 0]),
                                           num_classes=10), axis=1), 0, 1)
            # labels_n = np.apply_along_axis(np.nonzero, 1, labels).squeeze()

            neg_labels = np.random.permutation(labels)

            if self.noise_:
                noise_factor = 0.3
                data_noisy = data + noise_factor * \
                    np.random.normal(loc=0.0, scale=1.0, size=data.shape)
                data_noisy = np.clip(data_noisy, 0., 1.)

                return (data_noisy, labels)
            return (data, labels)
        else:
            if ret_bbox:
                return (data, labels)  # .reshape(self.batch_size_, -1)
            else:
                labels = np.clip(np.sum(
                    keras.utils.to_categorical(np.sort(labels[:, 0, :, 0]),
                                               num_classes=10), axis=1), 0, 1)
                return (data, labels)  # .reshape(self.batch_size_, -1)

    def __getitem__(self, index):
        return self.get_batch()

    def display_data(self, data, rec=None, fut=None,
                     fig=0, case_id=0, output_file=None, num_rows=1):
        output_file1 = None
        output_file2 = None

        if output_file is not None:
            name, ext = os.path.splitext(output_file)
            output_file1 = '%s_original%s' % (name, ext)
            output_file2 = '%s_recon%s' % (name, ext)

        # get data
        data = data[case_id].reshape(-1, self.image_size_, self.image_size_)
        # get reconstruction and future sequences if exist
        if rec is not None:
            rec = rec[case_id].reshape(-1, self.image_size_,
                                       self.image_size_)
            enc_seq_length = rec.shape[0]
        if fut is not None:
            fut = fut[case_id].reshape(-1, self.image_size_,
                                       self.image_size_)
            if rec is None:
                enc_seq_length = self.seq_length_ - fut.shape[0]
            else:
                assert enc_seq_length == self.seq_length_ - fut.shape[0]

        # create figure for original sequence
        plt.figure(2*fig, figsize=(20, num_rows*1))
        plt.clf()
        for i in range(self.seq_length_):
            plt.subplot(num_rows, self.seq_length_, i+1)
            plt.imshow(data[i, :, :],
                       cmap=plt.cm.gray, interpolation="nearest")
            plt.axis('off')
        plt.draw()
        if output_file1 is not None:
            print(output_file1)
            plt.savefig(output_file1, bbox_inches='tight')

        # create figure for reconstuction and future sequences
        if fut is not None or rec is not None:
            plt.figure(2*fig+1, figsize=(20, num_rows*1))
            plt.clf()
            for i in range(self.seq_length_):
                if rec is not None and i < enc_seq_length:
                    plt.subplot(num_rows, self.seq_length_, i + 1)
                    plt.imshow(rec[rec.shape[0] - i - 1, :, :],
                               cmap=plt.cm.gray, interpolation="nearest")
                if fut is not None and i >= enc_seq_length:
                    plt.subplot(num_rows, self.seq_length_, i + 1)
                    plt.imshow(fut[i - enc_seq_length, :, :],
                               cmap=plt.cm.gray, interpolation="nearest")
                plt.axis('off')

        plt.draw()
        if output_file2 is not None:
            print(output_file2)
            plt.savefig(output_file2, bbox_inches='tight')
        else:
            plt.pause(0.1)

    def on_epoch_end(self):
        np.random.seed(self.random_seed_)
        self.row_ = 0

        self.indices_ = np.arange(self.data_.shape[0])
        if self.shuffle_:
            np.random.shuffle(self.indices_)
