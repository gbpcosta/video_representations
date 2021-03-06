import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from glob import glob


class UCF101DataGenerator(keras.utils.Sequence):
    """Data Handler that loads UCF101 dataset on the fly."""

    DATASET_SIZE = {'train': 28747,
                    'test': 11213}

    def __init__(self, seq_length=16, batch_size=32,
                 image_size=64, shuffle=True,
                 frame_skip=1, to_gray=False,
                 split='train', random_seed=42, ae=False, noise=False,
                 data_path='UCF101'):
        if not os.path.exists(os.path.join(data_path, 'data', 'frames')):
            print('UCF101 folder must contain extracted frames for each video'
                  ' considering the following organisation: data/frames'
                  '/<class>/<video_name>/%05d.jpg.')

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
        self.class_dict = \
            pd.read_csv(os.path.join(data_path, 'classInd.txt'),
                        sep=' ', header=None, index_col=0)

        self.frames_path = os.path.join(data_path, 'data', 'frames')

        if self.split == 'train':
            columns = ['class_name', 'video_file', 'class_id']
        elif self.split == 'test':
            columns = ['class_name', 'video_file']
        else:
            raise NotImplementedError

        self.split_info = pd.DataFrame([], columns=columns)
        for ii in range(1, 4):
            aux = pd.read_csv(
                os.path.join(data_path,
                             '{0}list{1:02d}.txt'.format(self.split, ii)),
                sep=' |/', header=None, index_col=False)

            aux.columns = columns
            self.split_info = pd.concat([self.split_info, aux],
                                        axis=0, ignore_index=True)

        self.split_info['video_name'] = \
            self.split_info['video_file'].apply(lambda x: x.rstrip('.avi'))

        self.dataset_size_ = self.split_info.shape[0]
        # self.DATASET_SIZE[self.split]

        self.n_classes = self.class_dict.shape[0]
        self.n_labels = 1

        self.on_epoch_end()

    def get_batch_size(self):
        return self.batch_size_

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.dataset_size_ / self.batch_size_))

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
        if self.row_ + self.batch_size_ >= self.indices_.shape[0]:
            batch_info = self.split_info.iloc[self.indices_[self.row_:]]
            self.row_ = 0
        else:
            batch_info = self.split_info.iloc[
                self.indices_[self.row_:self.row_+self.batch_size_]]
            self.row_ += self.batch_size_

        # batch_folders_names = \
        #     batch_info.iloc[:, :3].apply(lambda x: '_'.join(x), axis=1)

        def get_last_frame_id(video_info):
            id = len(glob(os.path.join(self.frames_path,
                                       video_info['class_name'],
                                       video_info['video_name'],
                                       '*.jpg')))
            return id

        batch_first_frames = \
            batch_info.apply(
                lambda x: np.random.randint(
                    low=1,
                    high=get_last_frame_id(x)
                         - (self.frame_skip * self.seq_length_)),
                axis=1)

        # minibatch data
        if self.to_gray:
            n_channels = 1
        else:
            n_channels = 3
        data = np.zeros((batch_info.shape[0], self.seq_length_,
                         self.image_size_, self.image_size_, n_channels),
                        dtype=np.float32)
        labels = np.zeros((batch_info.shape[0], self.seq_length_,
                           self.n_classes), dtype=np.float32)

        for ii, sample in enumerate(batch_info.iterrows()):
            sample_frame_paths = \
                map(lambda x:
                    os.path.join(self.frames_path,
                                 batch_info['class_name'].iloc[ii],
                                 batch_info['video_name'].iloc[ii],
                                 '{:05d}.jpg'.format(x)),
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
                                   == batch_info['class_name'].iloc[ii])[0]] = 1

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
            plt.savefig(output_file2, bbox_inches='tight')
        else:
            plt.pause(0.1)

    def on_epoch_end(self):
        np.random.seed(self.random_seed_)
        self.row_ = 0

        self.indices_ = np.arange(self.split_info.shape[0])
        if self.shuffle_:
            np.random.shuffle(self.indices_)
