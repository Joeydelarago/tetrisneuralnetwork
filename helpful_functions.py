import matplotlib.pyplot as plt
import numpy as np

def basic_plot_time(title, x, color='blue'):
    """ Plots a vector x where the index represents time and the value represents the value at that time"""
    plt.figure()
    plt.title(title)
    plt.plot(range(len(x)), x, color=color)
    d = np.zeros(len(x))
    plt.fill_between(range(len(x)), x, color=color)
    plt.show()

def bar_plot_time(title, x, color='red'):
    """ Plots a vector x where the index represents time and the value represents the value at that time"""
    plt.figure()
    plt.title(title)
    plt.bar(range(len(x)), x, color=color)
    plt.show()

def reward_stats(rewards):
    """ Prints statistics about vector rewards where the index represents
        that episode and the value represents the value at the end of that episode
    """
    print("\n *** Reward Stats *** ")
    print("Average Reward: {}".format(sum(rewards)/len(rewards)))
    if len(rewards) >= 100:
        print("Average Reward Last 100 Episodes: {}".format(sum(rewards[-100:])/100))
    print("Total Reward: {}".format(sum(rewards)))

def step_stats(steps):
    """ Prints statistics about vector steps where the index represents
        the episode and the value represents the steps at the end of that episode
    """
    print("\n *** Step Stats *** ")
    print("Average Steps: {}".format(sum(steps)/len(steps)))
    if len(steps) >= 100:
        print("Average Steps Last 100 Episodes: {}".format(sum(steps[-100:])/100))

def episodes_until_reward_above_mean(reward):
    """ Prints the amount of steps it took until the reward was always above the total mean reward"""
    mean = np.mean(reward)
    for i in range(len(reward) - 1):
        if reward[len(reward) - i - 1] < mean:
            print("Episodes Until Reward Above Average Reward: {}".format(i))
            return
