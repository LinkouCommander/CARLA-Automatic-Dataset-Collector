import networkx as nx
from networkx.algorithms import community
import csv
import numpy as np
import matplotlib.pyplot as plt

# 步驟1：讀取CSV檔並解析其內容
def read_csv_to_edges(file_path):
    edges = []
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            # 每一列的第一個值是父節點，第二個值是子節點
            edges.append((row[0], row[1]))  # 添加邊到邊列表
    return edges

# 步驟2和3：將CSV中的邊轉換為NetworkX圖形
def create_graph_from_edges(edges):
    G = nx.Graph()
    G.add_edges_from(edges)
    return G

# CSV檔的路徑
csv_file_path = 'ast_tree.csv'

# 步驟1：讀取CSV檔
edges = read_csv_to_edges(csv_file_path)

# 步驟2和3：將CSV中的邊轉換為NetworkX圖形
graph = create_graph_from_edges(edges)

# 基於程式碼結構添加節點和邊

# 步驟2：計算基本指標
num_nodes = graph.number_of_nodes()
num_edges = graph.number_of_edges()

# 步驟3：計算進階指標
clustering_coefficient = nx.average_clustering(graph)
assortativity_coefficient = nx.degree_assortativity_coefficient(graph)
diameter = nx.diameter(graph)
# 您可能需要使用社群檢測算法計算模組性係數
# 您也需要將圖形轉換為鄰接矩陣，然後計算特徵值
# 使用 Louvain 演算法找到社群
partition = community.greedy_modularity_communities(graph)

# 計算模組性系數
modularity = community.modularity(graph, partition)

print("num_nodes:", num_nodes)
print("num_edges:", num_edges)
print("clustering_coefficient:", clustering_coefficient)
print("assortativity_coefficient:", assortativity_coefficient)
print("diameter:", diameter)
print("modularity:", modularity)

A = nx.adjacency_matrix(graph)

# 计算邻接矩阵的特征值
eigenvalues = np.linalg.eigvals(A.todense())

# 绘制特征值的分布图
plt.hist(eigenvalues, bins=20)
plt.title('Eigenvalue Distribution of Adjacency Matrix')
plt.xlabel('Eigenvalue')
plt.ylabel('Frequency')
plt.show()