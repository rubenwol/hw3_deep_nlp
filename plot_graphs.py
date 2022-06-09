import matplotlib.pyplot as plt
import json

def plot_graph(a_path, b_path, c_path, d_path, task='pos'):
    with open(a_path) as f:
        a = json.load(f)
    with open(b_path) as f:
        b = json.load(f)
    with open(c_path) as f:
        c = json.load(f)
    with open(d_path) as f:
        d = json.load(f)
    x = [i*5 for i in range(len(a))]
    plt.plot(x, a, label="rep a")
    plt.plot(x, b, label="rep b")
    plt.plot(x, c, label="rep c")
    plt.plot(x, d, label="rep d")
    plt.legend()
    plt.title(f'{task}')
    plt.savefig(f'plot/{task}', dpi=300)
    plt.close()


plot_graph('dev_acc/pos_rep_a.json', 'dev_acc/pos_rep_b.json', 'dev_acc/pos_rep_c.json', 'dev_acc/pos_rep_d.json')
plot_graph('dev_acc/ner_rep_a.json', 'dev_acc/ner_rep_b.json', 'dev_acc/ner_rep_c.json', 'dev_acc/ner_rep_d.json', task='ner')