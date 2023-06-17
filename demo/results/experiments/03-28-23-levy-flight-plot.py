import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    arr = np.array([32579, 42999, 10508, 196152, 7177, 100528, 67009, 29988, 16228, 118965, 13350, 25610, 38970, 34852, 27464, 7635,
     39929, 7044, 26471, 50269, 5118, 15675, 23362, 10585, 60948, 65898, 14103, 17277, 9342, 27949, 29378, 3637, 6385,
     12400, 32162, 15853, 46258, 42313, 25177, 47277, 44518, 6387, 37361, 11801, 24437, 38548, 15656, 4165, 18666,
     47541, 31319, 19250, 39512, 42281, 30276, 3607, 47785, 15792, 46466, 21687, 58315, 28083, 13006, 96224, 33347,
     8165, 49174, 16940, 44255, 1620, 12393, 10674, 10138, 77308, 34260, 9755, 13356, 27389, 10566, 2579, 25831, 11640,
     61508, 30185, 15710, 86999, 14859, 26461, 74055, 30610, 33016, 16570, 26786, 16461, 19021, 40821, 29294, 80193,
     26473, 15112])

    print(f"Average: {np.average(arr)}")
    print(f"Std. Dev.: {np.std(arr)}")

    plt.hist(arr, 10)
    plt.title("Timesteps to Goal - Single Levy Flight Agent - 100 trials")
    plt.xlabel("Time (Timesteps)")
    plt.ylabel("No. Trials")

    plt.show()