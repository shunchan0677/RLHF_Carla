import queue
import random
import socket
import time
from multiprocessing import Process

import gym
import gym_carla
import carla

import numpy as np
import pyglet

from a2c.common.atari_wrappers import wrap_deepmind
from scipy.ndimage import zoom
import cv2




# https://github.com/joschu/modular_rl/blob/master/modular_rl/running_stat.py
# http://www.johndcook.com/blog/standard_deviation/
class RunningStat(object):
    def __init__(self, shape=()):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM)/self._n
            self._S[...] = self._S + (x - oldM)*(x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        if self._n >= 2:
            return self._S/(self._n - 1)
        else:
            return np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


# Based on SimpleImageViewer in OpenAI gym
class Im(object):
    def __init__(self, display=None):
        self.window = None
        self.isopen = False
        self.display = display

    def imshow(self, arr):
        #print("arr.shape")
        #print(arr.shape)
        arr = np.flipud(arr)

        #cv2.imshow("Color Image", np.asarray(arr*255))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        #test = np.zeros((336,712,3))
        #test[:,:,0] = arr[:,:,0]
        #test[:,:,1] = arr[:,:,0]
        #test[:,:,2] = arr[:,:,0]
        #arr = test

        if self.window is None:
            height, width, _ = arr.shape
            self.window = pyglet.window.Window(
                width=width, height=height, display=self.display)
            self.width = width
            self.height = height
            self.isopen = True

        assert arr.shape == (self.height, self.width,3), \
            "You passed in an image with the wrong number shape"

        image = pyglet.image.ImageData(self.width, self.height,
                                       'RGB', arr.tobytes())
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0, 0)
        self.window.flip()

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()


class VideoRenderer:
    play_through_mode = 0
    restart_on_get_mode = 1

    def __init__(self, vid_queue, mode, zoom=1, playback_speed=1):
        assert mode == VideoRenderer.restart_on_get_mode or mode == VideoRenderer.play_through_mode
        self.mode = mode
        self.vid_queue = vid_queue
        self.zoom_factor = zoom
        self.playback_speed = playback_speed
        self.proc = Process(target=self.render)
        self.proc.start()

    def stop(self):
        self.proc.terminate()

    def render(self):
        v = Im()
        frames = self.vid_queue.get(block=True)
        t = 0
        while True:
            # Add a grey dot on the last line showing position
            width = frames[t].shape[1]
            fraction_played = t / len(frames)
            x = int(fraction_played * width)
            frames[t][-1][x] = 128

            frames[t] = np.asarray(frames[t]*255)
            zoomed_frame = zoom(frames[t], (self.zoom_factor, self.zoom_factor, 1))
            #print("zoomed_frame.shape")
            #print(zoomed_frame.shape)
            v.imshow(zoomed_frame)

            if self.mode == VideoRenderer.play_through_mode:
                # Wait until having finished playing the current
                # set of frames. Then, stop, and get the most
                # recent set of frames.
                t += self.playback_speed
                if t >= len(frames):
                    frames = self.get_queue_most_recent()
                    t = 0
                else:
                    time.sleep(1/60)
            elif self.mode == VideoRenderer.restart_on_get_mode:
                # Always try and get a new set of frames to show.
                # If there is a new set of frames on the queue,
                # restart playback with those frames immediately.
                # Otherwise, just keep looping with the current frames.
                try:
                    frames = self.vid_queue.get(block=False)
                    t = 0
                except queue.Empty:
                    t = (t + self.playback_speed) % len(frames)
                    time.sleep(1/60)

    def get_queue_most_recent(self):
        # Make sure we at least get something
        item = self.vid_queue.get(block=True)
        while True:
            try:
                item = self.vid_queue.get(block=True, timeout=0.1)
            except queue.Empty:
                break
        return item


def get_port_range(start_port, n_ports, random_stagger=False):
    # If multiple runs try and call this function at the same time,
    # the function could return the same port range.
    # To guard against this, automatically offset the port range.
    if random_stagger:
        start_port += random.randint(0, 20) * n_ports

    free_range_found = False
    while not free_range_found:
        ports = []
        for port_n in range(n_ports):
            port = start_port + port_n
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("127.0.0.1", port))
                ports.append(port)
            except socket.error as e:
                if e.errno == 98 or e.errno == 48:
                    print("Warning: port {} already in use".format(port))
                    break
                else:
                    raise e
            finally:
                s.close()
        if len(ports) < n_ports:
            # The last port we tried was in use
            # Try again, starting from the next port
            start_port = port + 1
        else:
            free_range_found = True

    return ports


def profile_memory(log_path, pid):
    import memory_profiler
    def profile():
        with open(log_path, 'w') as f:
            # timeout=99999 is necessary because for external processes,
            # memory_usage otherwise defaults to only returning a single sample
            # Note that even with interval=1, because memory_profiler only
            # flushes every 50 lines, we still have to wait 50 seconds before
            # updates.
            memory_profiler.memory_usage(pid, stream=f,
                                         timeout=99999, interval=1)
    p = Process(target=profile, daemon=True)
    p.start()
    return p


def batch_iter(data, batch_size, shuffle=False):
    idxs = list(range(len(data)))
    if shuffle:
        np.random.shuffle(idxs)  # in-place

    start_idx = 0
    end_idx = 0
    while end_idx < len(data):
        end_idx = start_idx + batch_size
        if end_idx > len(data):
            end_idx = len(data)

        batch_idxs = idxs[start_idx:end_idx]
        batch = []
        for idx in batch_idxs:
            batch.append(data[idx])

        yield batch
        start_idx += batch_size


class DiscountedGymWrapper(gym.Wrapper):
    def __init__(self, env, discount):
        super(DiscountedGymWrapper, self).__init__(env)
        self.discount = discount

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        discounted_reward = reward * self.discount
        return obs, discounted_reward, done, info


def make_env(env_id, seed=0):
    if env_id in ['carla-v0']:
        params = {
            'number_of_vehicles': 100,
            'number_of_walkers': 0,
            'display_size': 256,  # screen size of bird-eye render
            'max_past_step': 1,  # the number of past steps to draw
            'dt': 0.1,  # time interval between two frames
            'discrete': False,  # whether to use discrete control space
            'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
            'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
            'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
            'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
            'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
            'port': 2000,  # connection port
            'town': 'Town03',  # which town to simulate
            'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
            'max_time_episode': 1000,  # maximum timesteps per episode
            'max_waypt': 12,  # maximum number of waypoints
            'obs_range': 32,  # observation range (meter)
            'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
            'd_behind': 12,  # distance behind the ego vehicle (meter)
            'out_lane_thres': 2.0,  # threshold for out of lane
            'desired_speed': 8,  # desired speed (m/s)
            'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
            'display_route': True,  # whether to render the desired route
            'pixor_size': 64,  # size of the pixor labels
            'pixor': False,  # whether to output PIXOR observation
        }

        discount =1.0
        env = gym.make(env_id,params=params)
        py_env = DiscountedGymWrapper(
            env,
            discount=discount
        )
        return py_env #wrap_deepmind(env)
    if env_id in ['MovingDot-v0', 'MovingDotNoFrameskip-v0']:
        import gym_moving_dot
    env = gym.make(env_id)
    if env_id not in ['CarRacing-v2']:
        env.seed(seed)
        if env_id == 'EnduroNoFrameskip-v4':
            from enduro_wrapper import EnduroWrapper
            env = EnduroWrapper(env)
        return wrap_deepmind(env)
    return wrap_deepmind(env)
