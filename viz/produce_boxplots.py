
import pandas
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

replace_dict = {
    'spr': 'L_1',
    'pak': 'L_2',
    'rand': 'L_3',
    64: 'P_1',
    80: 'P_2',
    115: 'P_3',
    1: 'N_1',
    2: 'N_2',
    3: 'N_3',
    4: 'N_4',
    5: 'N_5'
}

proper_app_names = {
    'comd': 'CoMD',
    'ft': 'FT',
    'kripke': 'Kripke',
    'lu': 'LU',
    'mg': 'MG'
}

class ConfigTriple():
    def __init__(self, algorithm, bw_level, power_cap):
        self.algorithm = algorithm
        self.bw_level = bw_level
        self.power_cap = power_cap
    
    def __str__(self):
        alg_str = replace_dict[self.algorithm]
        bw_str = replace_dict[self.bw_level]
        pow_str = replace_dict[self.power_cap]

        return '<$%s$, $%s$, $%s$>' % (alg_str, bw_str, pow_str)


def gen_config_list(algorithms, bw_levels, power_caps):
    config_list = []
    for alg in algorithms:
        for bw in bw_levels:
            for pow_cap in power_caps:
                config_list.append(ConfigTriple(alg, bw, pow_cap))
    
    return config_list

def extract_values(data_frame, configuration):
    filtered_rows = data_frame[data_frame.algorithm == configuration.algorithm]
    filtered_rows = filtered_rows[filtered_rows.bw_level == configuration.bw_level]
    filtered_rows = filtered_rows[filtered_rows.power_cap == configuration.power_cap]

    return filtered_rows


def get_box_data(file_path):
    csv_panda = pandas.read_csv(file_path)

    csv_panda.runtime = csv_panda.runtime / (csv_panda.thread_count * csv_panda.node_count)
    csv_panda = csv_panda.drop(columns=["app", "ipath_0", "ipath_1", "node_count", "thread_count"])

    app_runtime_min = csv_panda.runtime.min()
    app_runtime_max = csv_panda.runtime.max()
    csv_panda.runtime = (csv_panda.runtime - app_runtime_min) / (app_runtime_max - app_runtime_min)

    x = []
    labels = []
    tick_locs = []
    major_space = [[]] * 4
    minor_space = [[]] * 3
    micro_space = [[]] * 2
    bounds_padding = [[]] * 2
    i = 1


    x.extend(bounds_padding)
    i += len(bounds_padding)

    for algorithm in ['spr', 'pak', 'rand']:
        for bw_level in [1, 2, 3, 4, 5]:
            #pad = ""
            for power_cap in [64, 80, 115]:
                config = ConfigTriple(algorithm, bw_level, power_cap)
                labels.append(power_cap)
                #labels.append(pad + str(config))
                #pad += "\n"

                result = extract_values(csv_panda, config)
                x.append(result.runtime)

                tick_locs.append(i)
                i += 1

                x.extend(micro_space)
                i += len(micro_space)
            x.extend(minor_space)
            i += len(minor_space)
        x.extend(major_space)
        i += len(major_space)
    
    while len(x[-1]) == 0:
        x.pop()
    
    x.extend(bounds_padding)

    return x, labels, tick_locs

def add_list_of_text(x, y, font_size, spacing, s_list):
    font_dict = {'fontsize': font_size}

    for i, s in enumerate(s_list):
        plt.text(x + spacing[i], y, s_list[i], fontdict=font_dict)


def place_color_grid(ax, color_x, color_y, color_step_size, color_width, color_height, color_list,
        text_x, text_y, text_step_size, text_list):

    for i in range(len(color_list)):
        ax.hlines(color_y + i * color_step_size, color_x, color_x+color_width, color=color_list[i], linewidth=color_height)
        plt.text(text_x, text_y + i * text_step_size, text_list[i])

def set_single_plot_text():
    x_pos = -10.0
    y_pos = -0.08
    y_step = -0.045
    plt.text(x_pos, y_pos, "Power", fontdict={'fontsize': 14})
    plt.text(x_pos - 5.0, y_pos + y_step, "Bandwidth", fontdict={'fontsize': 14})
    plt.text(x_pos - 3.0, y_pos + 2*y_step, "Location", fontdict={'fontsize': 14})

    bw_offset = 0.0
    bw_font_size = 12
    bw_spacing = [i*12 for i in range(5)]
    bw_list = [str(i) for i in range(1,6)]
    add_list_of_text(5.5, y_pos + y_step + bw_offset, bw_font_size, bw_spacing, bw_list)
    add_list_of_text(69.5, y_pos + y_step + bw_offset, bw_font_size, bw_spacing, bw_list)
    add_list_of_text(133.5, y_pos + y_step + bw_offset, bw_font_size, bw_spacing, bw_list)

    alg_offset = y_pos + 2*y_step
    alg_spacing = [0, 64, 128]
    add_list_of_text(26.5, alg_offset, 12, alg_spacing, ["Spread", "Packed", "Random"])

def produce_app_boxplot(app, output):
    plt.rc('font', size=10)
    plt.rcParams['figure.figsize'] = 18, 10

    x, labels, tick_locs = get_box_data("../data/%s.csv" % app)
    plt.title(proper_app_names[app], fontdict={'fontsize': 22})

    bp = plt.boxplot(x, widths=1.5, patch_artist=True)
    for patch in bp['boxes']:
        patch.set(facecolor=(0, 0, 0, 0.025))  


    plt.xticks(tick_locs, labels)

    set_single_plot_text()

    plt.savefig(output)
    plt.clf()


def produce_specific_configuration(algorithm, bw_level, power_cap, title, output):
    csv_files = ["comd", "mg", "ft", "lu", "kripke"]
    csv_files.sort()
    config = ConfigTriple(algorithm, bw_level, power_cap)
    labels = [proper_app_names[app] for app in csv_files]

    plt.rc('font', size=16)
    plt.rcParams['figure.figsize'] = 18, 10

    plt.title(title, fontdict={'fontsize': 22})

    x = []
    for index, csv_file_name in enumerate(csv_files):
        csv_panda = pandas.read_csv("../data/%s.csv" % csv_file_name)

        csv_panda.runtime = csv_panda.runtime / (csv_panda.thread_count * csv_panda.node_count)
        csv_panda = csv_panda.drop(columns=["app", "ipath_0", "ipath_1", "node_count", "thread_count"])

        app_runtime_min = csv_panda.runtime.min()
        app_runtime_max = csv_panda.runtime.max()
        csv_panda.runtime = (csv_panda.runtime - app_runtime_min) / (app_runtime_max - app_runtime_min)

        result = extract_values(csv_panda, config)
        x.append(result.runtime)
    
    bp = plt.boxplot(x, widths=0.3, patch_artist=True, labels=labels)
    for patch in bp['boxes']:
        patch.set(facecolor=(0, 0, 0, 0.025))
    
    axes = plt.gca()
    axes.set_ylim([0, 1.0])

    plt.savefig(output)
    #plt.show()
    plt.clf()

def produce_comd_mg_app_overlaid():
    csv_files = ["comd", "mg"]

    plt.rc('font', size=10)
    plt.rcParams['figure.figsize'] = 18, 10

    ax = plt.gca()

    alpha = 0.4
    colors = [
        ('darkblue', (0, 0, 1, alpha)),
        ('darkred', (1, 0, 0, alpha))
    ]

    plt.title("CoMD vs. MG", fontdict={'fontsize': 16})

    all_x = []
    bps = []
    for index, csv_file_name in enumerate(csv_files):
        x, labels, tick_locs = get_box_data("../data/%s.csv" % csv_file_name)
        all_x.append(x)

        edge_color, fill_color = colors[index]

        bp = ax.boxplot(x, widths=1.5, patch_artist=True)
        bps.append(bp)

        for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps', "means"]:
            plt.setp(bp[element], color=edge_color)
        
        for patch in bp['boxes']:
            patch.set(facecolor=fill_color)  
        
        plt.xticks(tick_locs, labels)
    
    ax.legend([bps[0]["boxes"][0], bps[1]["boxes"][0]], ['CoMD', 'MG'], loc='upper right')

    set_single_plot_text()


    plt.subplots_adjust(hspace=0.35)

    plt.savefig('boxplot_results/comd_mg_overlaid.png')
    #plt.show()
    plt.clf()



def main():
    # App specific charts
    for app in ["comd", "mg", "ft", "lu", "kripke"]:
        print("Starting %s..." % app)
        produce_app_boxplot(app, "boxplot_results/%s.png" % app)


    # Overlay
    print("Starting comd_mg overlay")
    produce_comd_mg_app_overlaid()

    # Configuration Specific charts
    print("Starting Traditional configuration")
    produce_specific_configuration('rand', 3, 115, 'Traditional Configuration', 'boxplot_results/traditional_config.png')

    print("Starting Power-limited configuration")
    produce_specific_configuration('rand', 3, 64, 'Power-Limited Configuration', 'boxplot_results/power_limited_config.png')
    
    print("Starting Network-limited configuration")
    produce_specific_configuration('rand', 1, 115, 'Network-Limited Configuration', 'boxplot_results/network_limited_config.png')
    
    print("Starting Topology-aware configuration")
    produce_specific_configuration('spr', 5, 115, 'Topology-Aware Configuration', 'boxplot_results/topology_aware_config.png')
    
    pass


if __name__ == "__main__":
    main()
