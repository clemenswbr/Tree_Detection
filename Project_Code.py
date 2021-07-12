#Load packages ----------------------------------------------------------------
from scipy import ndimage as ndi
from skimage import data, feature, filters
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import pyproj
import os

#Define path for pyproj library - Optional!
#pyproj.datadir.set_data_dir()

#Load data --------------------------------------------------------------------
data = gdal.Open('chm.tif')
array = np.array(data.GetRasterBand(1).ReadAsArray())

#Apply filters ----------------------------------------------------------------
#Maximum filter
data_max = ndi.maximum_filter(array, size=10, mode='constant')

#Gaussian filter
data_gauss = filters.gaussian(array, sigma=5)

#Image coordinates of local maxima
img_coords = feature.peak_local_max(data_gauss, threshold_rel=0.1)

#Plot results -----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(array, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('Original')

ax[1].imshow(data_max, cmap=plt.cm.afmhot_r)
ax[1].axis('off')
ax[1].set_title('Maximum filter')

ax[2].imshow(array, cmap=plt.cm.gray)
ax[2].autoscale(False)
ax[2].plot(img_coords[:, 1], img_coords[:, 0], 'r.')
ax[2].axis('off')
ax[2].set_title('Peak local max')

fig.tight_layout()
plt.show()

#Save Figure
fig.savefig('Output.png')

#Coordinates of max points ---------------------------------------------------- 
#Omit scientific notation
pd.set_option('display.float_format', lambda x: '%0.30f' % x)

#Convert image coordinates to real coordinates
map_layer = rasterio.open('chm.tif')    
coordinates = map_layer.xy(img_coords[:, 0], img_coords[:, 1])

#Convert coordinates to data frame
stack = np.column_stack((coordinates[0], coordinates[1]))
df_coordinates = pd.DataFrame(stack, columns=['X', 'Y'])

#Save to file -----------------------------------------------------------------
#Save coordinates to txt
df_coordinates[['X', 'Y']].to_csv('Tree_coordinates.txt', sep='\t')

#Save coordinates to xls
df_coordinates.to_excel('Tree_coordinates.xlsx')

#Write shapefile
points = df_coordinates.set_geometry(
    gpd.points_from_xy(df_coordinates['X'], df_coordinates['Y']), 
    inplace=False, crs='EPSG:5858')
points.to_file('Tree_coordinates_shp.shp')

