
import pandas
import matplotlib.pyplot as plt
import numpy as np

def produce_all_apps_scatter2():
    all_config_frame = pandas.read_csv("all_apps.csv")

    # Get the min and max of our y values 
    y_min = np.min(all_config_frame.score)
    y_max = np.max(all_config_frame.score)

    # Min-max normalize our plot
    all_config_frame.score = (all_config_frame.score - y_min) / (y_max - y_min)

    # 5 is abritary to scale our sizes
    all_config_frame.var_range_100 = 5 * all_config_frame.var_range_100

    color_map = {
        'CoMD': 'r',
        'FT': 'b',
        'LU': 'g',
        'MG': 'm',
        'Kripke': 'y',
        '64' : 'r',
        '80' : 'g',
        '115' : 'c'
    }



    # width x height
    plt.rcParams['figure.figsize'] = 15, 10
    plt.rc('font', size=18)

    ax = plt.gca()

    # for app in ["CoMD", "FT", "LU", "MG", "Kripke"]:
    #     # Filter by application
    #     filtered_frame = all_config_frame[all_config_frame.application == app.lower()]
        
    #     x = filtered_frame.config
    #     y = filtered_frame.score
    #     size = filtered_frame.var_range_100
    #     #color = [color_map[power_cap] * len(x)
    #     color = []
    #     for p in filtered_frame.power_cap:
    #         if p == 64:
    #             color.append('r')
    #         if p == 80:
    #             color.append('g')
    #         if p==115:
    #             color.append('b')
    #     print(color)
    #     ax.scatter(x, y, s=size, c=color, alpha=0.3)
    for app in [64, 80, 115]:
        # Filter by application
        filtered_frame = all_config_frame[all_config_frame.power_cap == app]       
        x = filtered_frame.config
        y = filtered_frame.score
        size = filtered_frame.var_range_100
        color = []
        for p in filtered_frame.power_cap:
            if p == 64:
                color.append('r')
            if p == 80:
                color.append('g')
            if p==115:
                color.append('b')
        print(color)
        ax.scatter(x, y, s=size, c=color, alpha=0.3)

    # 1.075 positions the legend. VERY SENSITIVE
#    ax.legend(('CoMD', 'FT', 'LU', 'MG', 'Kripke'), loc='upper right', ncol=5, bbox_to_anchor=(1, 1.075))
    ax.legend(('64W', '80W', '115W'), loc='upper right', ncol=3, bbox_to_anchor=(1, 1.075))

    # We want a tick at each interval
    ticks = [i for i in range(1, 46)]
    labels = []
    for i in range(1, 46):
        if i % 5 == 0 or i == 1:
            labels.append(i)  # Yes Label
        else:
            labels.append("")  # No label
    plt.xticks(ticks, labels)

    # Repeat the process to create black dots
    for app in ["CoMD", "FT", "LU", "MG", "Kripke"]:
        filtered_frame = all_config_frame[all_config_frame.application == app.lower()]
        x = filtered_frame.config
        y = filtered_frame.score   
        ax.scatter(x, y, s=2.0, c='k', alpha=1.0)

    # plt.xlabel("Configurations")
    plt.ylabel("Score")
    plt.savefig("test_bubble_3.png")
    plt.savefig("test_bubble_3.pdf")

    plt.clf()
    plt.cla()


def produce_all_apps_scatter():
    all_config_frame = pandas.read_csv("all_apps.csv")

    # Get the min and max of our y values 
    y_min = np.min(all_config_frame.score_minimum)
    y_max = np.max(all_config_frame.score_minimum)

    # Min-max normalize our plot
    all_config_frame.score_minimum = (all_config_frame.score_minimum - y_min) / (y_max - y_min)

    # 5 is abritary to scale our sizes
    all_config_frame.var_range_100 = 5 * all_config_frame.var_range_100

    color_map = {
        'CoMD': 'r',
        'FT': 'b',
        'LU': 'g',
        'MG': 'm',
        'Kripke': 'y'
    }



    # width x height
    plt.rcParams['figure.figsize'] = 15, 10
    plt.rc('font', size=18)

    ax = plt.gca()

    for app in ["CoMD", "FT", "LU", "MG", "Kripke"]:
        # Filter by application
        filtered_frame = all_config_frame[all_config_frame.application == app.lower()]

        x = filtered_frame.config
        y = filtered_frame.score_minimum
        size = filtered_frame.var_range_100

        color = [color_map[app]] * len(x)
        
        ax.scatter(x, y, s=size, c=color, alpha=0.3)

    # 1.075 positions the legend. VERY SENSITIVE
    ax.legend(('CoMD', 'FT', 'LU', 'MG', 'Kripke'), loc='upper right', ncol=5, bbox_to_anchor=(1, 1.075))


    # We want a tick at each interval
    ticks = [i for i in range(1, 46)]
    labels = []
    for i in range(1, 46):
        if i % 5 == 0 or i == 1:
            labels.append(i)  # Yes Label
        else:
            labels.append("")  # No label
    plt.xticks(ticks, labels)

    # Repeat the process to create black dots
    for app in ["CoMD", "FT", "LU", "MG", "Kripke"]:
        filtered_frame = all_config_frame[all_config_frame.application == app.lower()]
        x = filtered_frame.config
        y = filtered_frame.score_minimum   
        ax.scatter(x, y, s=2.0, c='k', alpha=1.0)

    plt.savefig("test_bubble_1.png")
    plt.savefig("test_bubble_1.pdf")

    plt.clf()
    plt.cla()

def produce_all_config_scatter():
    all_config_frame = pandas.read_csv("all_config.csv")

    # Get the min and max of our y values 
    y_min = np.min(all_config_frame.score)
    y_max = np.max(all_config_frame.score)

    # Min-max normalize our plot
    all_config_frame.score = (all_config_frame.score - y_min) / (y_max - y_min)
    
    # 10000 is abritary to scale our sizes
    all_config_frame.variance = 10000 * all_config_frame.variance

    color_map = {
        'CoMD': 'r',
        'FT': 'b',
        'LU': 'g',
        'MG': 'm',
        'Kripke': 'y'
    }

    # width x height
    plt.rcParams['figure.figsize'] = 15, 10
    plt.rc('font', size=18)

    ax = plt.gca()

    for app in ["CoMD", "FT", "LU", "MG", "Kripke"]:
        # Filter by application
        filtered_frame = all_config_frame[all_config_frame.application == app]
        x = filtered_frame.Id
        y = filtered_frame.score
        size = filtered_frame.variance

        color = [color_map[app]] * len(x)
        
        ax.scatter(x, y, s=size, c=color, alpha=0.3)

    # 1.075 positions the legend. VERY SENSITIVE
    ax.legend(('CoMD', 'FT', 'LU', 'MG', 'Kripke'), loc='upper right', ncol=5, bbox_to_anchor=(1, 1.075))


    # Repeat the process to create black dots
    for app in ["CoMD", "FT", "LU", "MG", "Kripke"]:
        filtered_frame = all_config_frame[all_config_frame.application == app]
        x = filtered_frame.Id
        y = filtered_frame.score
        ax.scatter(x, y, s=2.0, c='k', alpha=1.0)

    plt.xlabel("Configurations")
    plt.ylabel("Score")
    plt.savefig("test_bubble_2.pdf")
    plt.savefig("test_bubble_2.png")

    plt.clf()
    plt.cla()


if __name__ == "__main__":
    #produce_all_apps_scatter2()
    produce_all_config_scatter()
    
    
