import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data


rng = np.random.RandomState(2020)

def np_load_frame(filename, resize_height, resize_width, color=True):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename) if color else cv2.imread(filename, cv2.IMREAD_GRAYSCALE)[..., np.newaxis]
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized


class DataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, time_step=4, num_pred=1, color=True):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.setup()
        self.samples = self.get_all_samples()
        self.color = color

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.tif'))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])

    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            for i in range(len(self.videos[video_name]['frame'])-self._time_step):
                frames.append(self.videos[video_name]['frame'][i])
        return frames

    def __getitem__(self, index):
        video_name = self.samples[index].split('/')[-2]
        frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])
        
        batch = []
        for i in range(self._time_step+self._num_pred):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name+i-1], self._resize_height, self._resize_width, color=self.color)
            if self.transform is not None:
                batch.append(self.transform(image))

        return np.concatenate(batch, axis=0)
        
        
    def __len__(self):
        return len(self.samples)



class ChipDataLoader(data.Dataset):
    def __init__(self, video_folder, transform, img_size, win_size, step_size, time_step=4, num_pred=1, color=True):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._time_step = time_step
        self._num_pred = num_pred
        self.setup()
        self.samples = self.get_all_samples()
        self.color = color

        # set as an (x,y) tuple. Assume x==y if only an integer is provided
        self.img_size = (img_size, img_size) if type(img_size)==int else img_size
        self.win_size = (win_size, win_size) if type(win_size)==int else win_size
        self.step_size = (step_size, step_size) if type(step_size)==int else step_size

        # 
        self.num_x_steps = len(range(0, self.img_size[0]-self.win_size[0], self.step_size[0]))
        self.num_y_steps = len(range(0, self.img_size[1]-self.win_size[1], self.step_size[1]))

        
    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.tif'))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])

    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            for i in range(len(self.videos[video_name]['frame'])-self._time_step):
                frames.append(self.videos[video_name]['frame'][i])

        return frames

    def __getitem__(self, index):
        frame_indx = index//(self.num_x_steps*self.num_y_steps)
        x_step = (index % (self.num_x_steps*self.num_y_steps) ) // self.num_x_steps
        y_step = (index % (self.num_x_steps*self.num_y_steps) ) % self.num_x_steps
        x = x_step*self.step_size[0]
        y = y_step*self.step_size[1]

        video_name = self.samples[frame_indx].split('/')[-2]
        frame_name = int(self.samples[frame_indx].split('/')[-1].split('.')[-2])

        batch = []
        for i in range(self._time_step+self._num_pred):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name+i-1], self.img_size[1], self.img_size[0], color=self.color)


            if self.transform is not None:
                batch.append(self.transform(image)[:,y:y+self.win_size[1], x:x+self.win_size[0]])

        return np.concatenate(batch, axis=0)

    def __len__(self):
        return len(self.samples)*self.num_x_steps*self.num_y_steps
