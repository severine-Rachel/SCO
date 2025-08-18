import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def place_group(group, y, pos):
        n = len(group)
        for i, node in enumerate(group):
            pos[node] = (i - n/2, y)  # x, y
def data_to_matrix():
    df = pd.read_csv('H:\Documents\Quantique\CEMRACS\sessions_lecture\SCO\distance.csv', sep=',', header=None)
    df = df.iloc[1: , 1:]
    df = df.fillna(0)
    df = df.to_numpy()
    df = df.astype(int)
    df = df + df.T - np.diag(df.diagonal())
    print(df)

    df1 = pd.read_csv('H:\Documents\Quantique\CEMRACS\sessions_lecture\SCO\instance3.csv', sep=',', header=None)
    df1 = df1.iloc[1: , 1:]
    df1 = df1.fillna(0)
    df1 = df1.to_numpy()
    df1 = df1.astype(int)
    print(df1)
    return df, df1

def geographical_instance(df, df1):
    '''
    create a directed graph from matrix df1 for Geographical France
    '''
    
    node_names = {
        0: "Lille",
        1: "Paris",
        2: "Rennes",
        3: "Angers",
        4: "Lyon",
        5: "Bordeau",
        6: "Marseille"
    }
    
    rows, cols = np.nonzero(df1)                
    counts = df1[rows, cols].astype(int)    
    rows_rep = np.repeat(rows, counts)
    cols_rep = np.repeat(cols, counts)
    weights_rep = np.repeat(df[rows, cols], counts)
    
    edges = [(node_names[r], node_names[c], w) for r, c, w in zip(rows_rep, cols_rep, weights_rep)]
    gr = nx.MultiDiGraph()
    gr.add_edges_from(edges)
    pos = nx.spring_layout(gr) 

    edges = list(gr.edges(keys=True))
    
    for i, edge in enumerate(edges):
        rad = 0.3 if i % 2 == 0 else -0.3
        nx.draw_networkx_edges(
            gr, pos, edgelist=[edge],
            connectionstyle=f'arc3,rad={rad}',
            arrows=True
        )
    
    #Labels
    edge_labels = {}
    for u, v, key in gr.edges(keys=True):
        label = f"{key}"
        edge_labels[(u, v, key)] = label
    for i, ((u, v, key), label) in enumerate(edge_labels.items()):
        rad = 0.3 if i % 2 == 0 else -0.3
        x = (pos[u][0] + pos[v][0]) / 2
        y = (pos[u][1] + pos[v][1]) / 2
        plt.text(x, y + rad * 0.2, label, fontsize=9,color="green", ha='center', va='center')

    nx.draw_networkx_nodes(gr, pos, node_size=300, node_color='lightpink')
    nx.draw_networkx_labels(gr, pos,  font_size=10, font_color='blue')
    plt.show()
    return gr, df, df1, node_names

def Layers_instance(df):
    node_names = {
        0: "Lille",
        1: "Paris",
        2: "Rennes",
        3: "Angers",
        4: "Lyon",
        5: "Bordeaux",
        6: "Marseille",
        7: "Bordeaux2"
    }
    SourceNode = {6:"Marseille", 5:"Bordeaux", 4:"Lyon"}
    listSource = {v for v in SourceNode.values()}
    SinkNode = {0:"Lille", 2: "Rennes", 5:"Bordeaux2"}
    listSink = {v for v in SinkNode.values()}
    intermediate = {1:"Paris", 3:"Angers"}
    listIntermediate = {v for v in intermediate.values()}
    #construct a layer graph with SourceNodes to Node Paris and Angers to SinkNodes
    layer_graph = nx.DiGraph()
    for source in listSource:
        layer_graph.add_edge(source, "Paris")
        layer_graph.add_edge(source, "Angers")
    for sink in listSink:
        layer_graph.add_edge("Paris", sink)
        layer_graph.add_edge("Angers", sink)
    #layer graph to matrix
    #create a 7*7 matrix filed with 0
    layer_graph_matrix = np.zeros((8, 8))
    for edge in layer_graph.edges():
        # print(edge)
        a = (next((k for k, v in node_names.items() if v == edge[0]), None))
        b=(next((k for k, v in node_names.items() if v == edge[1]), None))
        layer_graph_matrix[a, b] = 1
    df_total = np.zeros((df.shape[0]+1, df.shape[1]+1))
    print("df_total shape = ", df_total.shape)
    for i in range(df.shape[0]):
        df_total[i, :-1] = df[i, :]
    print("df_total = ", df_total[:, -1])
    print("df = ", df[:, 5])
    df_8shape = df[:, 5]
    df_8shape = np.append(df_8shape, 0)
    print("df_8shape = ", df_8shape)
    df_total[:, -1] = df_8shape
    df_total[-1, :] = df_8shape
    print(df_total)
    print("layer matrice = ",layer_graph_matrix)
    rows, cols = np.where(layer_graph_matrix == 1)
    print(df, layer_graph_matrix)
    edge_labels = {(node_names[i], node_names[j]): df_total[i, j] for i, j in zip(rows, cols)}
    print("edge label = ", edge_labels)
    pos = {}
    # counts = df1[rows, cols].astype(int)
    
    place_group(listSource, 2, pos) 
    place_group(listIntermediate, 1, pos) 
    place_group(listSink, 0, pos) 
    node_colors = []
    for node in layer_graph.nodes():
        if node in listSource:
            print(node)
            node_colors.append('lightgreen')
        elif node in listSink:
            node_colors.append('lightcoral')
        else:
            node_colors.append('lightblue')
    # Affichage des labels
    nx.draw_networkx_edge_labels(layer_graph, pos, edge_labels=edge_labels)

    nx.draw(layer_graph, pos, with_labels=True, node_size=1000, node_color=node_colors, arrows=True)
    plt.show()

def instance_gen():
    df, df1 = data_to_matrix()
    geographical_instance(df, df1)
    Layers_instance(df)
    return geographical_instance(df, df1), Layers_instance(df)
'''
-faire une instance layer_graph
-rajouter les distance sur chaque arc depuis la matrice df
-transformer la matrice en qubo
-rajouter notre qubo dans l'ancien code d'A
'''
