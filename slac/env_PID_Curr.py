import imp
import numpy as np
import pyglet
from pyglet import image
import platform

from pyglet.gl import *

import PIL
import time
import os

import matplotlib
import time

homeDir = os.getcwd()
imgDir = homeDir + "/slac/trainingImagesPIDCurr/"
if not os.path.exists(imgDir):
    os.mkdir(imgDir)


class ArmEnv(object):
    viewer = None
    dt = .1    # refresh rate
    action_bound = [-1, 1]
    goal = {'x': 100., 'y': 100., 'l': 40}
    state_dim = 9
    action_dim = 2
    dampingK = 0.5
    goal_pos_radius = 200.

    def __init__(self):
        self.arm_info = np.zeros(
            2, dtype=[('l', np.float32), ('r', np.float32)])
        self.arm_info['l'] = 100        # 2 arms length
        self.arm_info['r'] = np.pi/6    # 2 angles information
        self.on_goal = 0
        # self.prev_finger_pos = np.array([0, 0])
        self.prev_finger_pos = np.array([174.88357544, 221.09895325])

        self.stepCount = 0
        self.velPast = np.array([0, 0])

    def step(self, action):
        done = False
        action = np.clip(action, *self.action_bound)
        self.arm_info['r'] += action * self.dt
        self.arm_info['r'] %= np.pi * 2    # normalize

        (a1l, a2l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r) = self.arm_info['r']  # radian, angle
        a1xy = np.array([200., 200.])    # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0]) / 400, (self.goal['y'] - a1xy_[1]) / 400]
        dist2 = [(self.goal['x'] - finger[0]) / 400, (self.goal['y'] - finger[1]) / 400]

        # Here we have the integral term (distance from target position)
        r = -np.sqrt(dist2[0]**2+dist2[1]**2)


        deltaPos = self.prev_finger_pos - finger
        self.velCurrent = deltaPos / 0.1

        deltaVel = self.velCurrent - self.velPast
        self.accCurrent = deltaVel / 0.1

        # Here we associate negative reward for high end effector velocity (prevent overshoot)
        if np.linalg.norm(self.velCurrent) > 80:
            rVel = -0.002*(np.linalg.norm(self.velCurrent) - 100)
            # print("rVel is: ", rVel)
            r+= rVel
        # Here we associate negative reward for high end effector accelleration (jerking)
        if np.linalg.norm(self.accCurrent) > 300:
            rAcc = -0.0002*(np.linalg.norm(self.accCurrent) - 300)
            # print("rAcc is: ", rAcc)
            r+= rAcc

        # print("\n")
        # time.sleep(0.1)


        # done and reward
        if self.goal['x'] - self.goal['l']/2 < finger[0] < self.goal['x'] + self.goal['l']/2:
            if self.goal['y'] - self.goal['l']/2 < finger[1] < self.goal['y'] + self.goal['l']/2:
                r += 1. #- dist/self.goal['l']
                self.on_goal += 1
                if self.on_goal > 50:
                    done = True
        else:
            self.on_goal = 0

        self.prev_finger_pos = finger
        self.velPast = self.velCurrent
        # state
        s = np.concatenate((a1xy_/200, finger/200, dist1 + dist2, [1. if self.on_goal else 0.]))

        self.stepCount += 1

        return s, r, done

    def reset(self):

        if self.stepCount <= 5 * 10 ** 5: # training the latent model, no assocation between action and block position
            self.goal['x'] = np.random.rand()*400.
            self.goal['y'] = np.random.rand()*400.

        elif 5 * 10 ** 5 < self.stepCount <= 8 * 10 ** 5: # intially training the planner (target a set radius away)
            x_center = np.random.choice([-1, 1])*np.random.rand()*150.
            self.goal['x'] = 200 + x_center
            # print("self.goal['x'] : ", self.goal['x'])
            self.goal['y'] = 200 + np.random.choice([-1, 1])*np.sqrt(150**2 - abs(x_center**2))
            # print("self.goal['y'] : ", self.goal['y'])

        elif 8 * 10 ** 5 < self.stepCount <= 10**6: # intially training the planner (target anywhere inside a set radius away)
            self.goal['x'] = 200 + np.random.choice([-1, 1])*np.random.rand()*150.
            self.goal['y'] = 200 + np.random.choice([-1, 1])*np.random.rand()*150.

        else: # finally the target can be anywhere
            self.goal['x'] = np.random.rand()*400.
            self.goal['y'] = np.random.rand()*400.

        # self.goal['x'] = np.random.rand()*400.
        # self.goal['y'] = np.random.rand()*400.

        self.arm_info['r'] = 2 * np.pi * np.random.rand(2)
        self.on_goal = 0
        (a1l, a2l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r) = self.arm_info['r']  # radian, angle
        a1xy = np.array([200., 200.])  # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, .get_image_data()y1)
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0])/400, (self.goal['y'] - a1xy_[1])/400]
        dist2 = [(self.goal['x'] - finger[0])/400, (self.goal['y'] - finger[1])/400]
        # state
        s = np.concatenate((a1xy_/200, finger/200, dist1 + dist2, [1. if self.on_goal else 0.]))

        # print("\n step_count is: ", self.stepCount)

        return s

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.arm_info, self.goal)        # matplotlib.pyplot.imsave(image_compress, render3)
        image = self.viewer.render()
        return image


class Viewer(pyglet.window.Window):
    bar_thc = 5

    def __init__(self, arm_info, goal, imgDir = imgDir):
        # self.glconfig = pyglet.gl.Config(sample_buffers=1, samples=8) # Windows
         # Linux
        self.imgDir = imgDir
        self.imgCount = 0

        vsync=False # To speed up training  
        if platform.system() == "Darwin":
            super(Viewer, self).__init__(width=400, height=400)

        elif platform.system() == "Linux":
            # self.glconfig = pyglet.gl.Config()
            # super(Viewer, self).__init__(config=self.glconfig, width=400, height=400, resizable=False, caption='AIM', vsync=vsync, style=pyglet.window.Window.WINDOW_STYLE_BORDERLESS)
            super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='AIM', vsync=vsync, style=pyglet.window.Window.WINDOW_STYLE_BORDERLESS)
            # NOTE: Made a modification here since pyglet.gl.Config was causing render issues

        elif platform.system() == "Windows":
            self.glconfig = pyglet.gl.Config(sample_buffers=1, samples=8)
            super(Viewer, self).__init__(config=self.glconfig, width=400, height=400, resizable=False, caption='AIM', vsync=vsync, style=pyglet.window.Window.WINDOW_STYLE_BORDERLESS)
        else:
            super(Viewer, self).__init__(width=400, height=400)
        

        pyglet.gl.glClearColor(0, 0, 0, 0)
        self.arm_info = arm_info
        self.goal_info = goal
        self.center_coord = np.array([200, 200])

        self.batch = pyglet.graphics.Batch()    # display whole batch at once
        self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [goal['x'] - goal['l'] / 2, goal['y'] - goal['l'] / 2, # location
                     goal['x'] - goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] - goal['l'] / 2]),
            ('c3B', (96, 96, 96) * 4))    # color

        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,                # location
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (255, 215, 0) * 4,),    # color
        )
        
        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,              # location
                     100, 160,
                     200, 160,        # matplotlib.pyplot.imsave(image_compress, render3)
                     200, 150]),
            ('c3B', (255, 215, 0) * 4,))    

    def render(self):
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip() 
        self.on_draw() 

        self.imgCount +=1

        image_save = str(self.imgDir + "img" + ".png")
        # image_compress = str(self.imgDir + "imgCompressed" + str(self.imgCount) + ".png")
        # use this line if you wish to save all images to imgDir

        pyglet.image.get_buffer_manager().get_color_buffer().save(image_save)

        render1 = PIL.Image.open(image_save)
        render2 = np.array(render1.resize((64,64), PIL.Image.ANTIALIAS)) # Compress image to reasonable size
        render3 = np.delete(render2, 3, 2) # delete 4th column of the 3rd dimension, data should be rgb
        render4 = np.moveaxis(render3, 2, 0) # (3, x, x) format for training

        return render4

    def on_draw(self):
        self.clear()
        self.batch.draw()
    
    def _update_arm(self):
        # update goal
        self.goal.vertices = (
            self.goal_info['x'] - self.goal_info['l']/2, self.goal_info['y'] - self.goal_info['l']/2,
            self.goal_info['x'] + self.goal_info['l']/2, self.goal_info['y'] - self.goal_info['l']/2,
            self.goal_info['x'] + self.goal_info['l']/2, self.goal_info['y'] + self.goal_info['l']/2,
            self.goal_info['x'] - self.goal_info['l']/2, self.goal_info['y'] + self.goal_info['l']/2)

        # update arm
        (a1l, a2l) = self.arm_info['l']     # radius, arm length
        (a1r, a2r) = self.arm_info['r']     # radian, angle
        a1xy = self.center_coord            # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy   # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r+a2r), np.sin(a1r+a2r)]) * a2l + a1xy_  # a2 end (x2, y2)

        a1tr, a2tr = np.pi / 2 - self.arm_info['r'][0], np.pi / 2 - self.arm_info['r'].sum()
        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc

        xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc

        self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))
        self.arm2.vertices = np.concatenate((xy11_, xy12_, xy21, xy22))

    # convert the mouse coordinate to goal's coordinate
    def on_mouse_motion(self, x, y, dx, dy):
        self.goal_info['x'] = x
        self.goal_info['y'] = y
