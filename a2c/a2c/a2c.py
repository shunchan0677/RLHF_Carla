import logging
import os.path as osp
import queue
import time

import cloudpickle
import easy_tf_log
import numpy as np
from numpy.testing import assert_equal
import tensorflow as tf

from a2c import logger
from a2c.a2c.utils import (cat_entropy, discount_with_dones,
                           find_trainable_variables, mse)
from a2c.common import explained_variance, set_global_seeds
from pref_db import Segment
import cv2
import copy
import matplotlib.pyplot as plt
import pyglet


#import tensorflow_probability as tfp


def gaussian_log_probs(mean, std, actions):
    var = std ** 2
    log_std = tf.math.log(std)
    log_probs = -0.5 * (((actions - mean) ** 2) / var + 2 * log_std + tf.math.log(2 * np.pi))
    return tf.reduce_sum(log_probs, axis=-1)

def sample_actions(mean, std):
    #std = tf.exp(log_std)
    noise = tf.random.normal(tf.shape(mean))
    sampled_actions = mean + std * noise
    return sampled_actions


def gaussian_entropy(std):
    return tf.reduce_sum(0.5 * (tf.math.log(2.0 * np.pi) + 2 * tf.math.log(std)) + 0.5, axis=-1)


class Model(object):
    def __init__(self,
                 policy,
                 ob_space,
                 ac_space,
                 nenvs,
                 nsteps,
                 nstack,
                 num_procs,
                 lr_scheduler,
                 ent_coef=0.01,
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 alpha=0.99,
                 epsilon=1e-5):
        config = tf.ConfigProto(
            allow_soft_placement=True,
            intra_op_parallelism_threads=num_procs,
            inter_op_parallelism_threads=num_procs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        nbatch = nenvs * nsteps 

        print("nbatch")
        print(nbatch)

        A = tf.placeholder(tf.float32, [nbatch* 2], name="A")
        ADV = tf.placeholder(tf.float32, [nbatch], name="B")
        R = tf.placeholder(tf.float32, [nbatch], name="C")
        LR = tf.placeholder(tf.float32, [])

        #print("ob_space")
        #print(ob_space["birdeye"])

        step_model = policy(
            sess, ob_space["birdeye"], ac_space, nenvs, 1, nstack, reuse=False, scope = "model")
        #print("ob_space")
        #print(ob_space)
        train_model = policy(
            sess, ob_space["birdeye"], ac_space, nenvs, nsteps, nstack, reuse=True, scope = "model")
        #print(ob_space)
        
        
        train_model_old = policy(
            sess, ob_space["birdeye"], ac_space, nenvs, nsteps, nstack, reuse=False, scope = "model2")
        
        old_model_output_stopped = tf.stop_gradient(train_model_old.pi)
        old_model_output_stopped2 = tf.stop_gradient(train_model_old.vf)

        mean_new = train_model.pi[:, 0:2]
        std_new = tf.math.softplus(train_model.pi[:, 2:4])+ 1e-6
        mean_old = train_model_old.pi[:, 0:2]
        std_old = tf.math.softplus(train_model_old.pi[:, 2:4])+ 1e-6

        #print(ob_space)

        sample_action = sample_actions(mean_new,std_new)

        new_log_probs = gaussian_log_probs(mean_new, std_new, sample_action)
        old_log_probs = gaussian_log_probs(mean_old, std_old, sample_action)

        #print(new_log_probs.shape)

        ratio = tf.exp(new_log_probs - old_log_probs)

        # Compute policy loss
        surrogate1 = ratio * ADV
        surrogate2 = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon) * ADV
        pg_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

        #print("pg_loss:=")
        #print(pg_loss)

        neglogpac = tf.constant(0)#tf.nn.sparse_softmax_cross_entropy_with_logits(
            #logits=train_model.pi, labels=A)
        #pg_loss = tf.constant(0)#tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R[0]))
        entropy_ = gaussian_entropy(std_new)
        entropy = tf.reduce_mean(entropy_) #tf.reduce_mean(cat_entropy(train_model.pi))
        loss = pg_loss - ent_coef*entropy +vf_loss * vf_coef

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(
            learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        #def update_old_model(train_model, old_model):
        #    old_model.set_weights(train_model.get_weights())

        def update_old_model(train_policy, old_policy):
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model")
            old_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model2")

            update_ops = []
            for old_var, new_var in zip(old_vars, train_vars):
                update_ops.append(old_var.assign(new_var))

            sess.run(update_ops)



        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            n_steps = len(obs)

            #print("ref actions")
            #print(actions)
            for _ in range(n_steps):
                cur_lr = lr_scheduler.value()
            td_map = {
                train_model.X: obs,
                train_model_old.X: obs,
                A: actions,
                ADV: advs,
                R: rewards,
                LR: cur_lr
            }
            if states != []:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train], td_map)
            return policy_loss, value_loss, policy_entropy, cur_lr

        self.train = train
        self.train_model = train_model
        self.train_model_old = train_model_old
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.sess = sess
        # Why var_list=params?
        # Otherwise we'll also save optimizer parameters,
        # which take up a /lot/ of space.
        # Why save_relative_paths=True?
        # So that the plain-text 'checkpoint' file written uses relative paths,
        # which seems to be needed in order to avoid confusing saver.restore()
        # when restoring from FloydHub runs.
        self.saver = tf.train.Saver(max_to_keep=0,var_list=params, save_relative_paths=True)
        tf.global_variables_initializer().run(session=sess)
        self.update_wight = update_old_model

    def load(self, ckpt_path):
        self.saver.restore(self.sess, ckpt_path)

    def save(self, ckpt_path, step_n):
        saved_path = self.saver.save(self.sess, ckpt_path, step_n)
        print("Saved policy checkpoint to '{}'".format(saved_path))


class Runner(object):
    def __init__(self,
                 env,
                 model,
                 nsteps,
                 nstack,
                 gamma,
                 gen_segments,
                 seg_pipe,
                 reward_predictor,
                 episode_vid_queue):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space["birdeye"].shape
        nenv = env.num_envs
        self.batch_ob_shape = (nenv * nsteps, nh, nw, nc * nstack)
        self.obs = np.zeros((nenv, nh, nw, nc * nstack), dtype=np.uint8)
        # The first stack of 4 frames: the first 3 frames are zeros,
        # with the last frame coming from env.reset().
        obs = env.reset()
        self.update_obs(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

        self.gen_segments = gen_segments
        self.segment = Segment()
        self.seg_pipe = seg_pipe

        self.orig_reward = [0 for _ in range(nenv)]
        self.reward_predictor = reward_predictor

        self.episode_frames = []
        self.episode_vid_queue = episode_vid_queue

    def rgb2gray(self, rgb):
        return rgb[:,:,1] #np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
    
    def display_gray_images(self, img):

        print(img.shape)
        # グレースケール画像をuint8型に変換し、範囲を[0, 255]にします
            
        # ウィンドウ名を設定し、画像を表示します
        window_name = f"Gray Image "
        cv2.imshow(window_name, img)
            
        # キーボード入力を待ちます（キーを押すと次の画像が表示されます）
        cv2.waitKey(0)

        # 全てのウィンドウを閉じます
        #cv2.destroyAllWindows()

    def display_grayscale_image(self, image_gray):

        # pyglet用の画像データに変換
        image_data = pyglet.image.ImageData(width=image_gray.shape[1], height=image_gray.shape[0], format="L", data=image_gray.tobytes(), pitch=image_gray.shape[1] * -1)

        # ウィンドウを作成
        window = pyglet.window.Window(width=image_data.width, height=image_data.height)

        # ウィンドウが閉じられるまでイベントループを実行
        @window.event
        def on_draw():
            window.clear()
            image_data.blit(0, 0)

        pyglet.app.run()

    
    def process_images(self, images_tuple):
        num_images = len(images_tuple)
        gray_images = np.empty((num_images, 96, 96, 3))

        #print("len")
        #print(images_tuple.shape)

        if(len(images_tuple.shape)==2):
            #for i, image in enumerate(images_tuple):
            #    gray_image = self.rgb2gray(image[0])
            #    gray_images[i, :, :, 0] = gray_image

            #gray_images = (gray_images * 255).astype(np.uint8)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!gray!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            #print(images_tuple[0]["birdeye"].shape)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!gray!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            #gray_images[:,:,:,0] = cv2.resize(images_tuple[0]["birdeye"], (96, 96), interpolation=cv2.INTER_AREA)
            gray_images[:,:,:,:] = cv2.resize(images_tuple[0]["birdeye"], (96, 96), interpolation=cv2.INTER_AREA)

            gray_images = (gray_images * 255).astype(np.uint8)
            print(gray_images.shape)

        #print("max")
        #print(np.max(gray_images))

        #for i, image in enumerate(images_tuple):
        #    gray_image2 = image[0]
        #    gray_images2[i, :, :, :] = gray_image2

        #self.display_gray_images(gray_images[-1,:,:,0])

        #img_to_display = (img.squeeze() * 255).astype(np.uint8)
        #print(gray_images[-1,:,:,0].shape)
        #self.display_grayscale_image(gray_images[-1,:,:,0])


        return gray_images



    def update_obs(self, obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce
        # IPC overhead
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!Update obs!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #print(self.obs.shape)
        #print(obs[0]["birdeye"].shape)
        #print(obs[1])
        self.obs = np.roll(self.obs, shift=-1, axis=3)

        self.obs[:, :, :, -3:] = obs[0]["birdeye"][:, :, :]
        #img = self.process_images(obs)
        #self.obs = obs[0]["birdeye"] #cv2.resize(img[0,:,:,:], (256, 256), interpolation=cv2.INTER_AREA)
        #for i in range(0,4):
        #    self.obs[:, :, :, 4*i:4*i+3] = obs[i]["birdeye"]
        #print(self.obs.shape)

    def update_segment_buffer(self, mb_obs, mb_rewards, mb_dones):
        # Segments are only generated from the first worker.
        # Empirically, this seems to work fine.
        e0_obs = (mb_obs[0] * 255).astype(np.uint8)
        e0_rew = mb_rewards[0]
        e0_dones = mb_dones[0]

        #self.display_grayscale_image(e0_obs[0,:,:,0])

        assert_equal(e0_obs.shape, (self.nsteps, 84, 84, 3*4))
        assert_equal(e0_rew.shape, (self.nsteps, ))
        assert_equal(e0_dones.shape, (self.nsteps, ))

        for step in range(self.nsteps):
            self.segment.append(np.copy(e0_obs[step]), np.copy(e0_rew[step]))
            if len(self.segment) == 25 or e0_dones[step]:
                while len(self.segment) < 25:
                    # Pad to 25 steps long so that all segments in the batch
                    # have the same length.
                    # Note that the reward predictor needs the full frame
                    # stack, so we send all frames.
                    self.segment.append(e0_obs[step], 0)
                self.segment.finalise()
                try:
                    self.seg_pipe.put(self.segment, block=False)
                except queue.Full:
                    # If the preference interface has a backlog of segments
                    # to deal with, don't stop training the agents. Just drop
                    # the segment and keep on going.
                    pass
                self.segment = Segment()

    def update_episode_frame_buffer(self, mb_obs, mb_dones):
        e0_obs = (mb_obs[0] * 255).astype(np.uint8)
        e0_dones = mb_dones[0]
        for step in range(self.nsteps):
            # Here we only need to send the last frame (the most recent one)
            # from the 4-frame stack, because we're just showing output to
            # the user.
            self.episode_frames.append(e0_obs[step, :, :, -3:])
            if e0_dones[step]:
                self.episode_vid_queue.put(self.episode_frames)
                self.episode_frames = []

    def run(self):
        nenvs = len(self.env.remotes)
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = \
            [], [], [], [], []
        mb_states = self.states

        # Run for nsteps steps in the environment
        for _ in range(self.nsteps):
            actions, values, states = self.model.step(self.obs, self.states,
                                                      self.dones)
            #print("actions:=")
            #print(actions)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            # len({obs, rewards, dones}) == nenvs
            obs, rewards, dones, _ = self.env.step(actions)
            self.states = states
            self.dones = dones

            #print("rewards")
            #print(rewards)


            #print("dones")
            #print(dones)

            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n] * 0
            # SubprocVecEnv automatically resets when done
            self.update_obs(obs)
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        # i.e. from nsteps, nenvs to nenvs, nsteps
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        # The first entry was just the init state of 'dones' (all False),
        # before we'd actually run any steps, so drop it.
        mb_dones = mb_dones[:, 1:]


        # Log original rewards
        for env_n, (rs, dones) in enumerate(zip(mb_rewards, mb_dones)):
            assert_equal(rs.shape, (self.nsteps, ))
            assert_equal(dones.shape, (self.nsteps, ))
            for step_n in range(self.nsteps):
                self.orig_reward[env_n] += rs[step_n]
                if dones[step_n]:
                    easy_tf_log.tflog(
                        "orig_reward_{}".format(env_n),
                        self.orig_reward[env_n])
                    self.orig_reward[env_n] = 0

        if self.env.env_id == 'MovingDotNoFrameskip-v0':
            # For MovingDot, reward depends on both current observation and
            # current action, so encode action in the observations.
            # (We only need to set this in the most recent frame,
            # because that's all that the reward predictor for MovingDot
            # uses.)
            mb_obs[:, :, 0, 0, -1] = mb_actions[:, :]

        # Generate segments
        # (For MovingDot, this has to happen _after_ we've encoded the action
        # in the observations.)
        mb_obs_tmp = copy.copy(mb_obs)
        if self.gen_segments:
            num_seq, num_imgs, height, width, num_channels = mb_obs.shape

            # 出力画像系列の形状を設定
            output_shape = (num_seq, num_imgs, 84, 84, 4*3)
            output_mb_obs = np.zeros(output_shape)

            for i in range(num_seq):
                for j in range(num_imgs):
                    img = mb_obs[i, j]
                    # カラーチャンネルを抽出
                    color_channels = [img[:, :, k:k+3] for k in range(0, num_channels, 3)]

                    # カラー画像をモノクロに変換し、リサイズ
                    mono_channels = []
                    for idx, color_channel in enumerate(color_channels):
                        gray = color_channel[:,:,:]#]cv2.cvtColor(color_channel, cv2.COLOR_BGR2GRAY)
                        #print(gray)
                        gray = (gray).astype(np.uint8)
                        resized_gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
                        #self.display_grayscale_image(gray)

                        mono_channels.append(resized_gray)

                    # モノクロ画像を4チャンネルに結合
                    #print(mono_channels.shape)
                    for p in range(4):
                        output_mb_obs[i, j,:,:,3*p:3*p+3] = mono_channels[p]#np.stack(mono_channels[p], axis=-1)

            mb_obs_tmp = copy.copy(mb_obs)
            mb_obs = output_mb_obs
            #print(mb_obs.shape)

            self.update_segment_buffer(mb_obs, mb_rewards, mb_dones)




        # Replace rewards with those from reward predictor
        # (Note that this also needs to be done _after_ we've encoded the
        # action.)
        logging.debug("Original rewards:\n%s", mb_rewards)
        if self.reward_predictor:
            assert_equal(mb_obs.shape, (nenvs, self.nsteps, 84, 84, 3*4))
            mb_obs_allenvs = mb_obs.reshape(nenvs * self.nsteps, 84, 84, 3*4)

            rewards_allenvs = self.reward_predictor.reward(mb_obs_allenvs)
            assert_equal(rewards_allenvs.shape, (nenvs * self.nsteps, ))
            mb_rewards = rewards_allenvs.reshape(nenvs, self.nsteps)
            assert_equal(mb_rewards.shape, (nenvs, self.nsteps))

            logging.debug("Predicted rewards:\n%s", mb_rewards)

        # Save frames for episode rendering
        if self.episode_vid_queue is not None:
            self.update_episode_frame_buffer(mb_obs, mb_dones)

        # Discount rewards
        #print(mb_obs.shape,self.batch_ob_shape)
        
        mb_obs = mb_obs_tmp.reshape(self.batch_ob_shape)

        last_values = self.model.value(self.obs, self.states,
                                       self.dones).tolist()
        

        #print(mb_obs.shape)
        #cv2.imshow("First Image", mb_obs[0,:,:,-1])
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(
                zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                # Make sure that the first iteration of the loop inside
                # discount_with_dones picks up 'value' as the initial
                # value of r
                rewards = discount_with_dones(rewards + [value],
                                              dones + [0],
                                              self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards

        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values


def learn(policy,
          env,
          seed,
          start_policy_training_pipe,
          ckpt_save_dir,
          lr_scheduler,
          nsteps=5,
          nstack=4,
          total_timesteps=int(80e6),
          vf_coef=0.5,
          ent_coef=0.01,
          max_grad_norm=0.5,
          epsilon=1e-5,
          alpha=0.99,
          gamma=0.99,
          log_interval=100,
          ckpt_save_interval=1000,
          ckpt_load_dir=None,
          gen_segments=False,
          seg_pipe=None,
          reward_predictor=None,
          episode_vid_queue=None):

    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes)  # HACK

    def make_model():
        return Model(
            policy=policy,
            ob_space=ob_space,
            ac_space=ac_space,
            nenvs=nenvs,
            nsteps=nsteps,
            nstack=nstack,
            num_procs=num_procs,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            lr_scheduler=lr_scheduler,
            alpha=alpha,
            epsilon=epsilon)

    with open(osp.join(ckpt_save_dir, 'make_model.pkl'), 'wb') as fh:
        fh.write(cloudpickle.dumps(make_model))

    print("Initialising policy...")
    if ckpt_load_dir is None:
        model = make_model()
    else:
        with open(osp.join(ckpt_load_dir, 'make_model.pkl'), 'rb') as fh:
            make_model = cloudpickle.loads(fh.read())
        model = make_model()

        ckpt_load_path = tf.train.latest_checkpoint(ckpt_load_dir)
        model.load(ckpt_load_path)
        print("Loaded policy from checkpoint '{}'".format(ckpt_load_path))

    ckpt_save_path = osp.join(ckpt_save_dir, 'policy.ckpt')

    runner = Runner(env=env,
                    model=model,
                    nsteps=nsteps,
                    nstack=nstack,
                    gamma=gamma,
                    gen_segments=gen_segments,
                    seg_pipe=seg_pipe,
                    reward_predictor=reward_predictor,
                    episode_vid_queue=episode_vid_queue)

    # nsteps: e.g. 5
    # nenvs: e.g. 16
    nbatch = nenvs * nsteps
    fps_tstart = time.time()
    fps_nsteps = 0

    print("Starting workers")

    # Before we're told to start training the policy itself,
    # just generate segments for the reward predictor to be trained with
    while True:
        runner.run()
        try:
            start_policy_training_pipe.get(block=False)
        except queue.Empty:
            continue
        else:
            break

    print("Starting policy training")

    for update in range(1, total_timesteps // nbatch + 1):
        # Run for nsteps
        obs, states, rewards, masks, actions, values = runner.run()

        policy_loss, value_loss, policy_entropy, cur_lr = model.train(
            obs, states, rewards, masks, actions, values)

        fps_nsteps += nbatch

        #print(policy_loss, value_loss, policy_entropy,actions)
        if(np.isnan(np.min(policy_loss))):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!loss is Nan!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            exit(0)

        if update % log_interval == 0 and update != 0:
            fps = fps_nsteps / (time.time() - fps_tstart)
            fps_nsteps = 0
            fps_tstart = time.time()

            print("Trained policy for {} time steps".format(update * nbatch))

            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("learning_rate", cur_lr)
            logger.dump_tabular()

        if update != 0 and update % (ckpt_save_interval * 10) == 0:
            model.save(ckpt_save_path, update)

            model.update_wight(model.train_model, model.train_model_old)

    model.save(ckpt_save_path, update)
