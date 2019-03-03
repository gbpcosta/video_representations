import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras


class KTHDataGenerator(keras.utils.Sequence):
    """Data Handler that loads KTH dataset on the fly."""

    DATASET_SIZE = {'train': 760,
                    'train_validation': 760+768,
                    'validation': 768,
                    'test': 863}

    def __init__(self, seq_length=16, batch_size=32,
                 image_size=64, shuffle=True,
                 frame_skip=1, to_gray=False,
                 split='train', random_seed=42, ae=False, noise=False,
                 kth_path='/store/gbpcosta/KTH'):
        if not os.path.exists(os.path.join(kth_path, 'data', 'frames')):
            print('KTH folder must contain extracted frames for each video'
                  ' considering the following organisation: data/frames'
                  '/<class>/<video_name>/%04d.jpg.')

        self.split = split
        self.to_gray = to_gray
        self.seq_length_ = seq_length
        self.batch_size_ = batch_size
        self.image_size_ = image_size
        self.frame_skip = frame_skip
        self.random_seed_ = random_seed
        self.shuffle_ = shuffle
        self.ae_ = ae
        self.noise_ = noise
        self.frames_path = os.path.join(kth_path, 'data', 'frames')

        if self.split == 'train_validation':
            self.dataset_size_ = self.DATASET_SIZE['train'] \
                + self.DATASET_SIZE['validation']
            aux = pd.read_csv(os.path.join(kth_path,
                                           'train.txt'),
                              sep='_|\t|-|,', header=None, index_col=False)

            aux2 = pd.read_csv(os.path.join(kth_path,
                                            'validation.txt'),
                               sep='_|\t|-|,', header=None, index_col=False)

            aux = pd.concat([aux, aux2], axis=0, ignore_index=True)
        else:
            self.dataset_size_ = self.DATASET_SIZE[self.split]
            aux = pd.read_csv(os.path.join(kth_path,
                                           '{}.txt'.format(self.split)),
                              sep='_|\t|-|,', header=None, index_col=False)

        aux.columns = ['person_id', 'class', 'scene_id',
                       'init_frame', 'end_frame',
                       'init_frame', 'end_frame',
                       'init_frame', 'end_frame',
                       'init_frame', 'end_frame']
        aux = pd.concat([aux.iloc[:, [0, 1, 2, 3, 4]],
                         aux.iloc[:, [0, 1, 2, 5, 6]],
                         aux.iloc[:, [0, 1, 2, 7, 8]],
                         aux.iloc[:, [0, 1, 2, 9, 10]]],
                        ignore_index=True)

        aux = aux.dropna()
        aux = aux.sort_values(by=['person_id', 'class', 'scene_id'])

        self.split_info = aux
        self.class_dict = self.split_info['class'].unique()
        self.n_classes = self.class_dict.shape[0]
        self.n_labels = 1

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

    def get_random_clip(self, video, begin, end):
        pass

    def preprocess_image(self, fname, bgrtogray=False, resize=None,
                         centered_crop=None, random_hflip=False):
        if not os.path.isfile(fname):
            print('{}: File does not exist!'.format(fname))

        image = cv2.imread(fname)

        if bgrtogray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if resize is not None:
            if isinstance(resize, (list, tuple)) and \
                    not isinstance(resize, str):
                image = cv2.resize(image, resize,
                                   interpolation=cv2.INTER_LINEAR)
            else:
                if image.shape[0] < image.shape[1]:
                    height = resize
                    width = int((height / image.shape[0]) * image.shape[1])
                    resize = (width, height)
                else:
                    width = resize
                    height = int((width / image.shape[1]) * image.shape[0])
                    resize = (width, height)
                image = cv2.resize(image, resize,
                                   interpolation=cv2.INTER_LINEAR)

        if centered_crop is not None:
            if isinstance(centered_crop, (list, tuple)) and \
                    not isinstance(centered_crop, str):
                image = image[((image.shape[0]-centered_crop[0])//2):
                                ((image.shape[0]+centered_crop[0])//2),
                              ((image.shape[1]-centered_crop[1])//2):
                                ((image.shape[1]+centered_crop[1])//2)]
            else:
                image = image[((image.shape[0]-centered_crop)//2):
                                ((image.shape[0]+centered_crop)//2),
                              ((image.shape[1]-centered_crop)//2):
                                ((image.shape[1]+centered_crop)//2)]

        if random_hflip:
            image = cv2.flip(image, 1)

        image = np.expand_dims(image, axis=-1)  # add channel dimension

        return image

    def get_batch(self, ret_bbox=False, verbose=False):
        # batch_info = self.split_info.sample(n=self.batch_size_,
        #                                     replace=False,
        #                                     random_state=self.random_seed_)
        if self.row_ + self.batch_size_ >= self.indices_.shape[0]:
            batch_info = self.split_info.iloc[self.indices_[self.row_:]]
            self.row_ = 0
        else:
            batch_info = self.split_info.iloc[
                self.indices_[self.row_:self.row_+self.batch_size_]]
            self.row_ += self.batch_size_

        batch_folders_names = \
            batch_info.iloc[:, :3].apply(lambda x: '_'.join(x), axis=1)

        # Check if every video in the batch has enough frames
        check_frame_availability = batch_info.apply(
            lambda x: (x[-1] - self.frame_skip * self.seq_length_ > x[-2]),
            axis=1)

        if not check_frame_availability.all():
            batch_info[~check_frame_availability].apply(
                lambda x: print('Insufficient number of frames in video'
                                ' {}'.format('_'.join(x[:3]))),
                axis=1)
            exit()

        batch_first_frames = \
            batch_info.iloc[:, 3:].apply(
                lambda x: np.random.randint(
                    low=x[0],
                    high=x[1] - (self.frame_skip * self.seq_length_)),
                axis=1)

        # minibatch data
        data = np.zeros((self.batch_size_, self.seq_length_,
                         self.image_size_, self.image_size_, 1),
                        dtype=np.float32)
        labels = np.zeros((self.batch_size_, self.seq_length_, self.n_classes),
                          dtype=np.float32)

        for ii, sample in enumerate(batch_info.iterrows()):
            sample_frame_paths = \
                map(lambda x:
                    os.path.join(self.frames_path,
                                 batch_info['class'].iloc[ii],
                                 batch_folders_names.iloc[ii],
                                 '{:04d}.jpg'.format(x)),
                    range(batch_first_frames.iloc[ii],
                          batch_first_frames.iloc[ii]
                          + (self.frame_skip * self.seq_length_),
                          self.frame_skip))

            # TODO check grayscale convertion method
            flip = np.random.choice([True, False])
            aux = np.array([
                self.preprocess_image(fname,
                                      bgrtogray=self.to_gray,
                                      resize=int(self.image_size_ * 1.1428),
                                      # imagenet ratio (256 - 224)
                                      centered_crop=self.image_size_,
                                      random_hflip=flip)
                for fname in sample_frame_paths])

            data[ii] = aux
            labels[ii, :, np.where(self.class_dict
                                   == batch_info['class'].iloc[ii])] = 1

        if self.ae_ is True:
            labels = labels[:, 0]

            if self.noise_:
                noise_factor = 0.3
                data_noisy = data + noise_factor * \
                    np.random.normal(loc=0.0, scale=1.0, size=data.shape)
                data_noisy = np.clip(data_noisy, 0., 1.)

                return (data_noisy, labels)
            return (data, labels)
        else:
            labels = labels[:, 0]
            return (data, labels)

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

        self.indices_ = np.arange(self.split_info.shape[0])
        if self.shuffle_:
            np.random.shuffle(self.indices_)
