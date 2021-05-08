import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data
from utils import is_power_of_2, nearest_power_of_2

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
    def __init__(self, video_folder, transform, resize_height, resize_width, time_step=4, num_pred=1, color=True, ext=".jpg"):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.setup()
        self.frames = self.get_all_samples()
        self.color = color
        self.ext = ext

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, "*"+self.ext))
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
        video_name = self.frames[index].split('/')[-2]
        frame_name = int(self.frames[index].split('/')[-1].split('.')[-2])
        
        batch = []
        for i in range(self._time_step+self._num_pred):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name+i-1], self._resize_height, self._resize_width, color=self.color)
            if self.transform is not None:
                batch.append(self.transform(image))

        return np.concatenate(batch, axis=0)
        
    def __len__(self):
        return len(self.frames)



class ChipDataLoader(data.Dataset):
    def __init__(self, video_folder, transform, img_size, win_size, step_size, time_step=4, num_pred=1, color=True, ext=".jpg"):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._time_step = time_step
        self._num_pred = num_pred
        self.color = color
        self.ext = ext

        # set as an (x,y) tuple. Assume x==y if only an integer is provided
        self.img_size = (img_size, img_size) if type(img_size)==int else img_size
        self.win_size = (win_size, win_size) if type(win_size)==int else win_size
        self.step_size = (step_size, step_size) if type(step_size)==int else step_size

        # ensure that image size is power of 2
        if not is_power_of_2(self.img_size[0]) or not is_power_of_2(self.img_size[1]):
            raise f"Image dimensions must be a power of 2. current={self.img_size}"

        # verify that the win_size is a power of 2 in both dimensions and correct if it isn't
        if not is_power_of_2(self.win_size[0]):
            np2 = nearest_power_of_2(self.win_size[0])
            print(f"win_size_x {self.win_size[0]} is not a power of 2. modifying to {np2}")
            self.win_size = (np2, self.win_size[1])

        if not is_power_of_2(self.win_size[1]):
            np2 = nearest_power_of_2(self.win_size[1])
            print(f"win_size_y {self.win_size[1]} is not a power of 2. modifying to {np2}")
            self.win_size = (self.win_size[0], np2)
        
        # if the window size is greater than or equal to img_size, make equal to img_size
        if self.win_size[0] >= self.img_size[0]:
            self.win_size = (self.img_size[0], self.win_size[1])
            self.num_x_steps = 1
        else:
            self.num_x_steps = len(range(0, self.img_size[0], self.step_size[0]))

        if self.win_size[1] >= self.img_size[1]:
            self.win_size = (self.win_size[0], self.img_size[1])
            self.num_y_steps = 1
        else:
            self.num_y_steps = len(range(0, self.img_size[1], self.step_size[1]))

        self.setup()

        # since this DataLoader loads a sequence of frames, this will get all of
        # the frames that are the start of a sequence
        self.seq_starts = self.get_start_frames()
        
        # get all of the frames that are the end of a sequence
        # (These are the frames that are being re-created by the model)
        self.seq_stops = self.get_stop_frames()


    def setup(self):
        videos = sorted(glob.glob(os.path.join(self.dir, '*')))
        for video in videos:
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frames'] = sorted(glob.glob(os.path.join(video, "*"+self.ext)))
            self.videos[video_name]['length'] = len(self.videos[video_name]['frames'])

    def get_start_frames(self):
        frames = []
        for video in self.videos:
            video_name = video.split('/')[-1]
            for i in range(len(self.videos[video_name]['frames'])-self._time_step):
                frames.append(self.videos[video_name]['frames'][i])
        return frames

    def get_stop_frames(self):
        frames = []
        for video in self.videos:
            video_name = video.split('/')[-1]
            for i in range(self._time_step, len(self.videos[video_name]['frames'])):
                frames.append(self.videos[video_name]['frames'][i])
        return frames

    def get_video(self, index):
        frame_indx = index//(self.num_x_steps*self.num_y_steps)
        video_name = self.seq_starts[frame_indx].split('/')[-2]
        return video_name

    def get_frame(self, index):
        frame_index = index//(self.num_x_steps*self.num_y_steps)
        #video_name = self.frames[seq_index].split('/')[-2]

        # the frame index represents the starting point of a sequency of frames but we want the last
        #frame_num = int(self.frames[seq_index].split('/')[-1].split('.')[-2])+self._time_step
        
        #frame_path = self.videos[video_name]['frames'][frame_num-1]
        return self.seq_stops[frame_index], np_load_frame(self.seq_stops[frame_index], self.img_size[1], self.img_size[0], color=self.color)

    def get_chip_indices(self, index):
        frame_indx = index//(self.num_x_steps*self.num_y_steps)
        x_step = (index % (self.num_x_steps*self.num_y_steps) ) // self.num_x_steps
        y_step = (index % (self.num_x_steps*self.num_y_steps) ) % self.num_x_steps
        x = x_step*self.step_size[0]
        y = y_step*self.step_size[1]
        return x,x+self.step_size[0],y,y+self.step_size[1]

    def chips_per_frame(self):
        """ The number of chips per frame is equal to the self.num_x_steps * self.num_y_steps"""
        return self.num_x_steps * self.num_y_steps

    def __getitem__(self, index):
        frame_indx = index//(self.num_x_steps*self.num_y_steps)
        x_step = (index % (self.num_x_steps*self.num_y_steps) ) // self.num_x_steps
        y_step = (index % (self.num_x_steps*self.num_y_steps) ) % self.num_x_steps
        x = x_step*self.step_size[0]
        y = y_step*self.step_size[1]

        video_name = self.seq_starts[frame_indx].split('/')[-2]
        frame_num = int(self.seq_starts[frame_indx].split('/')[-1].split('.')[-2])

        batch = []
        for i in range(self._time_step+self._num_pred):
            image = np_load_frame(self.videos[video_name]['frames'][frame_num+i-1], self.img_size[1], self.img_size[0], color=self.color)

            if self.transform is not None:
                batch.append(self.transform(image)[:,y:y+self.win_size[1], x:x+self.win_size[0]])

        return np.concatenate(batch, axis=0)

    def __len__(self):
        return len(self.seq_starts)*self.num_x_steps*self.num_y_steps
