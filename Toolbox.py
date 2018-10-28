#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:33:46 2018

@author: yao_we
"""
import sys

def readGeotiff(raster):
    if raster is None:
        print("Unable to open INPUT.tif")    
        sys.exit(1)
    
    # Projection
    #raster.GetProjection()
    
    # Dimensions
    x_size = raster.RasterXSize
    y_size = raster.RasterYSize
    
    # print("[ Raster band count ]: ", raster.RasterCount)

    # print("[ Getting band ]: ", band)
    # raster_band = raster.GetRasterBand(band)   
    
    # Read raster data as numeric array from GDAL Dataset
    # raster_band_array = raster_band.ReadAsArray()    
    
    
    # print(raster.GetMetadata())
        
    raster_array = raster.ReadAsArray()
    # print(raster_array.shape)
    # print(" Type of raster array:\n ", raster_array)

    return raster_array
        
#    stats = srcband.GetStatistics( True, True )
#    if stats is None:
#        continue
#        
#    print("[ STATS ] = Minimum=%.3f, Maximum=%.3f, Mean=%.3f, StdDev=%.3f" % ( \
#          stats[0], stats[1], stats[2], stats[3]))


# Normalizd Difference Vegetation Index (NDVI)
# NDVI = (NIR - Rot) / (NIR + Rot)
# pixelArray is already a vector which stores the band values for a pixel location
def NDVI_3by3(pixelArray):
    
    ###### Computing NDVI of the image.
    # Nominator
    nom_ = pixelArray[3] - pixelArray[2]
    
    # Denominator
    denom_ = pixelArray[3] + pixelArray[2]
    
    # NDVI
    NDVI = nom_ / (denom_ + 0.00001)
    
    return NDVI
