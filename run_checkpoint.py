#!/usr/bin/env python

"""
Run a trained checkpoint to see what the agent is actually doing in the
environment.
"""

import argparse
import os.path as osp
import time
from collections import deque

import cloudpickle
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from utils import make_env
import cv2


def main():
    args = parse_args()

    env = make_env(args.env)
    #print(env.continuous)
    model = get_model(args.policy_ckpt_dir)
    if args.reward_predictor_ckpt_dir:
        reward_predictor = get_reward_predictor(args.reward_predictor_ckpt_dir)
    else:
        reward_predictor = None

    run_agent(env, model, reward_predictor, args.frame_interval_ms)


def run_agent(env, model, reward_predictor, frame_interval_ms):
    nenvs = 1
    nstack = int(model.step_model.X.shape[-1])
    nh, nw, nc = env.observation_space["birdeye"].shape
    nc = 1
    obs = np.zeros((nenvs, nh, nw, nc * nstack), dtype=np.uint8)
    model_nenvs = int(model.step_model.X.shape[0])
    states = model.initial_state
    number_of_episode = 10
    reward_list = []
    step_list = []
    if reward_predictor:
        value_graph = ValueGraph()
    while number_of_episode > 0:
        raw_obs = env.reset()
        update_obs(obs, raw_obs, nc)
        episode_reward = 0
        done = False
        i = 0
        while not done:
            i += 1
            model_obs = np.vstack([obs] * model_nenvs)
            actions, _, states = model.step(model_obs, states, [done])
            action = actions[0]
            output = env.step(action)
            raw_obs=output[0]
            reward = output[1]
            done = output[2]
            print(episode_reward, done, action, i)
            if(i > 8000):
                done = True
            obs = update_obs(obs, raw_obs, nc)
            episode_reward += reward
            #env.render()
            if reward_predictor is not None:
                predicted_reward = reward_predictor.reward(obs)
                # reward_predictor.reward returns reward for each frame in the
                # supplied batch. We only supplied one frame, so get the reward
                # for that frame.
                value_graph.append(predicted_reward[0])
            time.sleep(frame_interval_ms * 1e-3)
        print("Episode reward:", episode_reward)
        print("Episode step:", i)
        reward_list.append(episode_reward)
        step_list.append(i)
        number_of_episode += -1
    print(reward_list)
    print(step_list)


def update_obs(obs, raw_obs, nc):
    #obs = np.roll(obs, shift=-nc, axis=3)
    #obs[:, :, :, -nc:] = process_images(raw_obs)
    #print(raw_obs)
    obs = np.roll(obs, shift=-1, axis=3)

    obs[:, :, :, -3:] = raw_obs["birdeye"][:, :, :]
    return obs

def display_gray_images(images_array):

        print(images_array.shape)
        for i, img in enumerate(images_array):
            # グレースケール画像をuint8型に変換し、範囲を[0, 255]にします
            img_to_display = (img.squeeze() * 255).astype(np.uint8)
            
            # ウィンドウ名を設定し、画像を表示します
            window_name = f"Gray Image "
            cv2.imshow(window_name, img_to_display)
            
            # キーボード入力を待ちます（キーを押すと次の画像が表示されます）
            #cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 全てのウィンドウを閉じます
        #cv2.destroyAllWindows()


def process_images(images_tuple):
        num_images = len(images_tuple)
        gray_images = np.empty((1, 96, 96, 1))

        #print("len")
        #print(len(np.array(images_tuple[0]).shape))
        if(len(np.array(images_tuple[0]).shape) == 3):
            gray_images[0,:,:,0] = images_tuple[0][:,:,1]
            display_gray_images(gray_images)
            return gray_images
        else:
            gray_images[0,:,:,0] = images_tuple[:,:,1]
            display_gray_images(gray_images)
            return gray_images

        if(True):
            for i, image in enumerate(images_tuple):
                gray_image = image[:,:,1]
                gray_images[i, :, :, 0] = gray_image
        else:
            gray_images[:,:,:,0] = images_tuple[:,:,:,1]

        #for i, image in enumerate(images_tuple):
        #    gray_image2 = image[0]
        #    gray_images2[i, :, :, :] = gray_image2
        #self.display_gray_images(gray_images)

        return gray_images


def get_reward_predictor(ckpt_dir):
    with open(osp.join(ckpt_dir, 'make_reward_predictor.pkl'), 'rb') as fh:
        make_reward_predictor = cloudpickle.loads(fh.read())
    cluster_dict = {'a2c': ['localhost:2200']}
    print("Initialising reward predictor...")
    reward_predictor = make_reward_predictor(name='a2c', cluster_dict=cluster_dict)
    reward_predictor.init_network(ckpt_dir)
    return reward_predictor


def get_model(ckpt_dir):
    model_file = osp.join(ckpt_dir, 'make_model.pkl')
    with open(model_file, 'rb') as fh:
        make_model = cloudpickle.loads(fh.read())
    print("Initialising policy...")
    model = make_model()
    ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
    print("Loading checkpoint...")
    model.load(ckpt_file)
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("env")
    parser.add_argument("policy_ckpt_dir")
    parser.add_argument("--reward_predictor_ckpt_dir")
    parser.add_argument("--frame_interval_ms", type=float, default=0.)
    args = parser.parse_args()
    return args


class ValueGraph:
    def __init__(self):
        n_values = 100
        self.data = deque(maxlen=n_values)

        self.fig, self.ax = plt.subplots()
        self.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        self.fig.set_size_inches(4, 2)
        self.ax.set_xlim([0, n_values - 1])
        self.ax.grid(axis='y')  # Draw a line at 0 reward
        self.y_min = float('inf')
        self.y_max = -float('inf')
        self.line, = self.ax.plot([], [])

        self.fig.show()
        self.fig.canvas.draw()

    def append(self, value):
        self.data.append(value)

        self.y_min = min(self.y_min, min(self.data))
        self.y_max = max(self.y_max, max(self.data))
        self.ax.set_ylim([self.y_min, self.y_max])
        self.ax.set_yticks([self.y_min, 0, self.y_max])
        plt.tight_layout()

        ydata = list(self.data)
        xdata = list(range(len(self.data)))
        self.line.set_data(xdata, ydata)

        self.ax.draw_artist(self.line)
        self.fig.canvas.draw()


if __name__ == '__main__':
    main()
