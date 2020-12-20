
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time

from eda_prep import load_dataframe_from_yelp_2


def create_edge_list(df):
    df['friends_list'] = df['friends'].apply(lambda x: x.split(', '))
    df.drop('friends', inplace=True, axis=1)
    df['edge_list'] = [[(x, z) for z in y] for x, y in zip(df['user_id'],
                                                           df['friends_list'])]
    return df


def create_graph(num_rec):
    Graph = nx.Graph()
    source = df['edge_list'].iloc[:num_rec]
    for record in source:
        Graph.add_edges_from(record)
    return Graph


if __name__ == "__main__":
    # 1968703 - Max records for full dataset.
    num_records = 100
    query = f'''
            SELECT user_id, friends
            FROM user_friends_10k
            LIMIT {num_records}
            ;
            '''
    st = time.perf_counter()
    df = load_dataframe_from_yelp_2(query)
    ft = time.perf_counter()
    print(f'Loaded the Dataframe with {num_records} in {(ft - st):.4f} seconds')
    ft = time.perf_counter()
    df = create_edge_list(df)
    ft = time.perf_counter()
    print(f'Created the edge_list column with {num_records} records in {(ft - st):.4f} seconds')
    st = time.perf_counter()
    Graph = create_graph(num_records)
    ft = time.perf_counter()
    print(f'Built the Graph from {num_records} records in {(ft - st):.4f} seconds')

    # Create metrics/features
    st1 = time.perf_counter()
    st = time.perf_counter()
    deg_cent = nx.degree_centrality(Graph)
    ft = time.perf_counter()
    print(f'Calculated deg_cent in {(ft - st):.4f} seconds')
    st = time.perf_counter()
    bet_cent = nx.betweenness_centrality(Graph)
    ft = time.perf_counter()
    print(f'Calculated bet_cent in {(ft - st):.4f} seconds')
    st = time.perf_counter()
    try:
        eig_cent = nx.eigenvector_centrality(Graph)
    except nx.PowerIterationFailedConvergence:
        print("Eigenvector Centrality Failed")
    ft = time.perf_counter()
    print(f'Calculated eig_cent in {(ft - st):.4f} seconds')
    st = time.perf_counter()
    close_cent = nx.closeness_centrality(Graph)
    ft = time.perf_counter()
    print(f'Calculated close_cent in {(ft - st):.4f} seconds')
    st = time.perf_counter()
    load_cent = nx.load_centrality(Graph)
    ft = time.perf_counter()
    print(f'Calculated load_cent in {(ft - st):.4f} seconds')
    st = time.perf_counter()
    vote_rank = nx.voterank(Graph)
    ft = time.perf_counter()
    print(f'Calculated vote_rank in {(ft - st):.4f} seconds')
    st = time.perf_counter()
    page_rank = nx.pagerank(Graph)
    ft = time.perf_counter()
    print(f'Calculated page_rank in {(ft - st):.4f} seconds')
    ft1 = time.perf_counter()
    print(f'Calculated ALL the metrics in {(ft1 - st1):.4f} seconds')

    # Visualize
    st = time.perf_counter()
    pos = nx.spring_layout(Graph)
    node_color = [20000.0 * Graph.degree(v) for v in Graph]
    node_size = [v * 10000 for v in bet_cent.values()]
    plt.figure(figsize=(20, 20))
    nx.draw_networkx(Graph, pos=pos, with_labels=False,
                     node_color=node_color,
                     node_size=node_size)
    plt.axis('off')
    ft = time.perf_counter()
    print(f'Made the visual in {(ft - st):.4f} seconds')
    plt.show()
