import numpy as np
import random
import pandas as pd

class KMeans:
    def __init__(self, df:pd.DataFrame, k:int) -> None:
        self.df = df
        self.data = df = df.iloc[: , 1:].to_numpy()
        self.k = k

        ## Initlialize the list of clusters. Clusters are represented as np.arrays
        self.clusters_list = [np.array([])]*k
        self.cluster_dic = {} #contains the index of the point and the index cluster it belongs to
        self.centroids = np.array([])

        print("Graph succesfuly initialized !")
    


    def get_cluster(self, i: int) -> np.array:
        """Returns the array containing the points of cluster i, i in 0, k-1"""
        return self.clusters_list[i]

    def get_point_cluster(self, p:int) -> np.array:
        """Returns the cluster the point p belongs to"""
        ## p is the index of the point as stored in self.data !!!!
        cluster_number = self.cluster_dic[p]
        return self.clusters_list[cluster_number]

    def get_centroid(self, i:int) -> np.array:
        """Returns the current centroid of cluster i"""
        return self.centroids[i]

    ### Evaluation
    
    def SSE(self) -> int:
        """Computes the SSE of the current clustering"""
        try:
            sse = sum(sum((np.linalg.norm(self.get_centroid(i)- p))**2 for p in self.get_cluster(i)) for i in range(self.k))
            return sse
        except:
            "SSE computing error : one or more clusters are empty ! "
        

        
    
    ### Computing centroids

    def uniform_random_centroids(self):
        n = int(self.data.shape[0])
        choosen_centroids = np.random.choice(self.data.shape[0], self.k, replace=False)
        self.centroids = self.data[choosen_centroids, :]
        print("Centroids have been randomly choosen succesfully ! ")

    def create_clusters(self):
        ## Reset cluster list
        self.clusters_list = [np.array([])]*self.k
        try:
            ## Compute minimal distances 
            for p in range(len(self.data)):
                point = self.data[p]
                d_min = 1e99
                closest_centroid = -1
                for i in range(self.k):
                    d = np.linalg.norm(point - self.centroids[i])
                    if d < d_min:
                        d_min = d
                        closest_centroid = i
                self.cluster_dic[p] = closest_centroid
                self.clusters_list[i] = np.append(self.clusters_list[i], point)
            
        except:
            print("Error : clusters centroids are not defined !")
    
    def update_centroids(self):
        """"""
        for i in range(self.k):
            self.centroids[i] = np.mean(self.clusters_list[i])




    

        

df = pd.read_csv("data.csv")
k = KMeans(df, 8)

k.uniform_random_centroids()
k.create_clusters()
k.update_centroids()
print(k.centroids)    