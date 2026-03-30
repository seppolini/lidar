'''
point cloud data is stored as a 2D matrix
each row has 3 values i.e. the x, y, z value for a point

Project has to be submitted to github in the private folder assigned to you
Readme file should have the numerical values as described in each task
Create a folder to store the images as described in the tasks.

Try to create commits and version for each task.

'''
#%%
import matplotlib
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path #added to resolve filepath



#%% utility functions
def show_cloud(points_plt):
    ax = plt.axes(projection='3d')
    ax.scatter(points_plt[:,0], points_plt[:,1], points_plt[:,2], s=0.01) #Z value is [:,2]
    plt.show()

def show_scatter(x,y):
    plt.scatter(x, y)
    plt.show()

def get_ground_level(pcd):
    return 62  #change to 61 based on histogram analyze for both data set


#adding pointer to file in root
HERE = Path(__file__).resolve().parent

name_dataset ="dataset2.npy" #Adjust here and pick out file name for automatic name in plot


#%% read file containing point cloud data
pcd = np.load(HERE / name_dataset) #adjust pointer. 

###name_dataset = pcd.name  #pick out file name for automatic name in plot



pcd.shape

#%% show downsampled data in external window
##%matplotlib qt
show_cloud(pcd)
#show_cloud(pcd[::10]) # keep every 10th point

#%% remove ground plane

'''
Task 1 (3)
find the best value for the ground level
One way to do it is useing a histogram 
np.histogram

update the function get_ground_level() with your changes

For both the datasets
Report the ground level in the readme file in your github project
Add the histogram plots to your project readme
'''

#historgram
z_values = pcd[:, 2]
hist, bin_edges = np.histogram(z_values, bins=50) #experiment with bins

max_idx = np.argmax(hist)
z_min = bin_edges[max_idx]
z_max = bin_edges[max_idx + 1]
max_count = hist[max_idx]

plt.figure(figsize=(8, 5))
plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge')


#function for plotting value of highest bin 
plt.axvspan(z_min, z_max, color='red', alpha=0.3)
plt.text(
z_min,
max_count,
f"Max Bin Value:\n[{z_min:.2f}, {z_max:.2f}]",
ha='left',
va='bottom',
color='red'

)



plt.xlabel("z-value")
plt.ylabel("Count of specific Value (Z)")
plt.title(name_dataset)
plt.grid(True)
plt.show()



est_ground_level = get_ground_level(pcd)
print(est_ground_level)

pcd_above_ground = pcd[pcd[:,2] > est_ground_level] 
#%%
pcd_above_ground.shape

#%% side view
show_cloud(pcd_above_ground)


# %%
unoptimal_eps = 10
# find the elbow
clustering = DBSCAN(eps = unoptimal_eps, min_samples=5).fit(pcd_above_ground)

#%%
clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, clusters)]

# %%
# Plotting resulting clusters
plt.figure(figsize=(10,10))
plt.scatter(pcd_above_ground[:,0], 
            pcd_above_ground[:,1],
            c=clustering.labels_,
            cmap=matplotlib.colors.ListedColormap(colors),
            s=2)


plt.title('DBSCAN: %d clusters' % clusters,fontsize=20)
plt.xlabel('x axis',fontsize=14)
plt.ylabel('y axis',fontsize=14)
plt.show()


#%%
'''
Task 2 (+1)

Find an optimized value for eps.
Plot the elbow and extract the optimal value from the plot
Apply DBSCAN again with the new eps value and confirm visually that clusters are proper

https://www.analyticsvidhya.com/blog/2020/09/how-dbscan-clustering-works/
https://machinelearningknowledge.ai/tutorial-for-dbscan-clustering-in-python-sklearn/

For both the datasets
Report the optimal value of eps in the Readme to your github project
Add the elbow plots to your github project Readme
Add the cluster plots to your github project Readme
'''




#%%
'''
Task 3 (+1)

Find the largest cluster, since that should be the catenary, 
beware of the noise cluster.

Use the x,y span for the clusters to find the largest cluster

For both the datasets
Report min(x), min(y), max(x), max(y) for the catenary cluster in the Readme of your github project
Add the plot of the catenary cluster to the readme

'''
