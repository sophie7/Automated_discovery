#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 19:29:17 2018

@author: yao_we
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 17:20:31 2018

@author: yao_we

ImageViewer
"""

from PyQt5 import QtCore,QtGui
from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap, QColor, QPen
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel,
        QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy)


from osgeo import gdal
import sys
import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, YearLocator, DateFormatter, drange
import csv
import datetime as dt
from matplotlib.backends.backend_pdf import PdfPages


import Toolbox
from clustering import *


class ImageViewer(QMainWindow):
    def __init__(self):
        super(ImageViewer, self).__init__()

        self.scaleFactor = 0.0

        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)  #####
        self.imageLabel.setScaledContents(True)

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.setCentralWidget(self.scrollArea)

        self.createActions()
        self.createMenus()

        self.setWindowTitle("Time Series Analysis")
        self.scrollWidth = 1515 #(3030/2)
        self.scrollHeight = 737 #(1474/2)
        self.resize(1515, 737)
        
        self.x = 0
        self.y = 0
        
        
    def mousePressEvent(self, QMouseEvent):
        print(QMouseEvent.pos())

    def mouseReleaseEvent(self, QMouseEvent):
        cursor =QtGui.QCursor()
        print(cursor.pos())

    def open(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                QDir.currentPath())
        if fileName:
            image = QImage(fileName)
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return
            
            raster = gdal.Open(fileName)
            data = Toolbox.readGeotiff(raster)           
            

            # self.imageLabel.setPixmap(QPixmap.fromImage(data))
            self.imageLabel.setPixmap(QPixmap.fromImage(image))
            self.scaleFactor = 1.0

            self.fitToWindowAct.setEnabled(True)
            self.updateActions()

            if not self.fitToWindowAct.isChecked():
                self.imageLabel.adjustSize()


    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)       

        
        if not fitToWindow:
            self.normalSize()

        self.updateActions()

    def about(self):
        QMessageBox.about(self, "About Time Series Viewer",
                "<p>The <b>Image Viewer</b> example shows how to combine "
                "QLabel and QScrollArea to display an image. QLabel is "
                "typically used for displaying text, but it can also display "
                "an image. QScrollArea provides a scrolling view around "
                "another widget. If the child widget exceeds the size of the "
                "frame, QScrollArea automatically provides scroll bars.</p>"
                "<p>The example demonstrates how QLabel's ability to scale "
                "its contents (QLabel.scaledContents), and QScrollArea's "
                "ability to automatically resize its contents "
                "(QScrollArea.widgetResizable), can be used to implement "
                "zooming and scaling features.</p>"
                "<p>In addition the example shows how to use QPainter to "
                "print an image.</p>")
       
    # View only one band
    def view(self):
        fileNameDir = "Data/1999_2017_landsat_time_series/roi/cut/fileNames.txt"
        if 'fileNameDir' in locals():
            pass
        else:
            raise AssertionError("FILE_NAME_DIR is not given!")
            
        with open(fileNameDir, "r") as file:
            imageFileNames = file.readlines()   # imageFiles is a list
            file.close()        
        
        for fileName in imageFileNames:
            fileName = "Data/1999_2017_landsat_time_series/roi/cut/" + fileName[:-1] + ".tif"

            image = QImage(fileName)
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                                        "Cannot load %s." % fileName)
                return                

            self.imageLabel.setPixmap(QPixmap.fromImage(image))            
            self.scaleFactor = 1.0
             
            #self.printAct.setEnabled(True)
            self.fitToWindowAct.setEnabled(True)
            self.updateActions()
             
            if not self.fitToWindowAct.isChecked():
                self.imageLabel.adjustSize()            
            app.processEvents()
            
            time.sleep(0.001)           

       
        """
        Timeline example:
            QTimeLine timeline = new QTimeLine(1000)
            timeLine.setFrameRange(0,100)
            connect(timeline, SIGNAL(frameChanged(int)), yourobj, SLOT(yourobjslot(int)))
            timeline.start()                
        """   

    # View RGB three bands
    def bandView(self):
        fileNameDir = "Data/1999_2017_landsat_time_series/roi/cut/fileNames.txt"
        if 'fileNameDir' in locals():
            pass
        else:
            raise AssertionError("FILE_NAME_DIR is not given!")
            
        with open(fileNameDir, "r") as file:
            imageFileNames = file.readlines()   # imageFiles is a list
            file.close()        
        
        for fileName in imageFileNames:
            fileName = "Data/1999_2017_landsat_time_series/roi/cut/" + fileName[:-1] + ".tif"
            
            raster = gdal.Open(fileName)
            data = Toolbox.readGeotiff(raster)  
            
            # If data is more than one band!
            if len(data.shape) != 3:
                continue      
            if data.shape[0] < 3:
                continue
            x_size = data.shape[2]
            y_size = data.shape[1] 

    
                  
            '''
            ###########
            img = QImage(fileName)
            ptr = img.bits()
            ptr.setsize(img.byteCount())

            ## copy the data out as a string
            strData = ptr.asstring()
            ## get a read-only buffer to access the data
            buf = memoryview(ptr)
            ## view the data as a read-only numpy array            
            arr = np.frombuffer(buf, dtype=np.ubyte).reshape(img.height(), img.width(), 4)
            ## view the data as a writable numpy array
            arr = np.asarray(ptr).reshape(img.height(), img.width(), 4)
            ############            
            '''
            
            # Choose the band:
                    
            ### Bug solved - data type should be declared earlier!
            imageNew = np.zeros((y_size, x_size, 4), dtype=np.uint8) 
            imageNew[:, :, 0] = data[2, :, :]
            imageNew[:, :, 1] = data[1, :, :]
            imageNew[:, :, 2] = data[0, :, :]              
#           if data.shape[0] == 6:
#                imageNew[:, :, 0] = data[2, :, :]
#                imageNew[:, :, 1] = data[1, :, :]
#                imageNew[:, :, 2] = data[0, :, :]
#            else:
#                imageNew[:, :, 0] = data[3, :, :]
#                imageNew[:, :, 1] = data[3, :, :]
#                imageNew[:, :, 2] = data[3, :, :]
            imageNew[:, :, 3] = 255            

            qimg = QImage(imageNew, x_size, y_size, QImage.Format_RGB32)
            
            self.imageLabel.setPixmap(QPixmap.fromImage(qimg))        
              
            self.fitToWindow()        
            self.updateActions()      
            app.processEvents()
            

            # Change the title
            QMainWindow.setWindowTitle(self, fileName) 
 
            time.sleep(0.2)          
    
    def getPos(self , event):         
        pos = self.imageLabel.mapFromParent(event.pos())
        self.x = pos.x()
        self.y = pos.y()           
            
    def viewNDVI(self):        
        fileNameDir = "Data/1999_2017_landsat_time_series/roi/cut/fileNames.txt"
        if 'fileNameDir' in locals():
            pass
        else:
            raise AssertionError("FILE_NAME_DIR is not given!")
            
        with open(fileNameDir, "r") as file:
            imageFileNames = file.readlines()   # imageFiles is a list
            file.close()   
            
        
        NDVI_seq = []
        fig, ax = plt.subplots()
        
        for fileName in imageFileNames:
            fileNameNew = "Data/1999_2017_landsat_time_series/roi/cut/" + fileName[:-1] + ".tif"
            
            raster = gdal.Open(fileNameNew)
            data = Toolbox.readGeotiff(raster)  
            # close dataset
            raster = None
            
            # If data is more than one band!
            if len(data.shape) != 3:
                continue      
            if data.shape[0] < 3:
                continue
            x_size = data.shape[2]
            y_size = data.shape[1] 

    
                  
            '''
            ###########
            img = QImage(fileName)
            ptr = img.bits()
            ptr.setsize(img.byteCount())

            ## copy the data out as a string
            strData = ptr.asstring()
            ## get a read-only buffer to access the data
            buf = memoryview(ptr)
            ## view the data as a read-only numpy array            
            arr = np.frombuffer(buf, dtype=np.ubyte).reshape(img.height(), img.width(), 4)
            ## view the data as a writable numpy array
            arr = np.asarray(ptr).reshape(img.height(), img.width(), 4)
            ############            
            '''
            
            # Choose the band:
                    
            ### Bug solved - data type should be declared earlier!
            imageNew = np.zeros((y_size, x_size, 4), dtype=np.uint8) 
            imageNew[:, :, 0] = data[2, :, :]
            imageNew[:, :, 1] = data[1, :, :]
            imageNew[:, :, 2] = data[0, :, :]              
#           if data.shape[0] == 6:
#                imageNew[:, :, 0] = data[2, :, :]
#                imageNew[:, :, 1] = data[1, :, :]
#                imageNew[:, :, 2] = data[0, :, :]
#            else:
#                imageNew[:, :, 0] = data[3, :, :]
#                imageNew[:, :, 1] = data[3, :, :]
#                imageNew[:, :, 2] = data[3, :, :]
            imageNew[:, :, 3] = 255            

            qimg = QImage(imageNew, x_size, y_size, QImage.Format_RGB32)
            
            self.imageLabel.setPixmap(QPixmap.fromImage(qimg))        
              
            self.fitToWindow()
            self.scaleFactor = 2.0
            self.updateActions()      
            app.processEvents()
            

            # Change the title
            QMainWindow.setWindowTitle(self, fileName)            
           

            self.imageLabel.setObjectName("image")
            self.imageLabel.mousePressEvent = self.getPos
            
            actualX = self.x * self.scaleFactor
            actualY = self.y * self.scaleFactor
#            qcolor = QColor(qimg.pixel(actualX, actualY))
#            # BGR
#            red = qcolor.red()
#            green = qcolor.green()
#            blue = qcolor.blue()
#
#            print("Scalefactor is: ", self.scaleFactor)
            print("Cursor locations x & y:", actualX, actualY)
#            print("Red, green, blue values: ", red, green, blue)
            
            sentinel_No = fileName[9]
            print("sentinel number is:", int(sentinel_No))

            actualX = int(actualX)
            actualY = int(actualY)

            ######
            # Search for similar temporal patterns
            actualX = 2495
            actualY = 310
            ######
            
            if data.shape[0] >= 4:
                if sentinel_No == 8:
                    NDVI = Toolbox.NDVI_3by3_8(data[:, actualY, actualX])
                    NDVI_seq.append(NDVI)  
                else:         
                    NDVI = Toolbox.NDVI_3by3_7(data[:, actualY, actualX])
                    NDVI_seq.append(NDVI)     
            
            ax.cla()
            plt.plot(NDVI_seq)
            plt.ylim(0, 250)
            plt_name = str(actualX) + str(actualY) + '.png'
            plt.savefig(plt_name)
            # ax.set_title(fileName)   
         
            
            # plt.pause(0.1)          
            # time.sleep(0.02)    
            
            del data, imageNew, qimg
        
    # Show kMeans overlay -- not finished!  
    def kMeansOverlay(self):    
        ###########################################################
        # Only use band 1 to read x,y pixel locations
        pixelInfo = pd.read_csv("Clustering/PixelInfo_Band1.csv", sep='\t', encoding='utf-8')
        indicesPd = pd.read_csv('Clustering/Clustering_results/label_7classes.tsv', sep='\t', encoding='utf-8')
        indices = indicesPd['0'].astype(int)
        indiceCluster = (indices == 7)
        pixelCluster = pixelInfo[indiceCluster]        


#        ''' Tensorflow clustering '''        
#        n_features = 220
#        n_clusters = 10
#        n_samples_per_cluster = int(len(pixel_info) / n_clusters)
#        seed = 700
#        embiggen_factor = 70
#        
#        
#        Band_1 = tf.convert_to_tensor(seqArray, np.float32)
#        initial_centroids = choose_random_centroids(Band_1, n_clusters)
#        nearest_indices = assign_to_nearest(Band_1, initial_centroids)
#        updated_centroids = update_centroids(Band_1, nearest_indices, n_clusters)
#        
#        model = tf.global_variables_initializer()
#        with tf.Session() as session:
#            sample_values = session.run(Band_1)
#            updated_centroid_value = session.run(updated_centroids)
#            nearest_indices_value = session.run(nearest_indices)
         
        
        
        ############################################################
        fileNameDir = "Data/1999_2017_landsat_time_series/roi/cut/fileNames_220.txt"
        if 'fileNameDir' in locals():
            pass
        else:
            raise AssertionError("FILE_NAME_DIR is not given!")
            
        with open(fileNameDir, "r") as file:
            imageFileNames = file.readlines()   # imageFiles is a list
            file.close()        
        
        
        ############# Mask Qimage ###################################
        imageWidth = 3030
        imageHeight = 1474
        bytesPerPixel = 4  # 4 for RGBA
        maskData = np.zeros((imageHeight, imageWidth, 4), dtype=np.uint8)
        # maskData[:, :, 3] = 100
        
        for idx, row in pixelCluster.iterrows():
            pixel_x = row['x_pixel']
            pixel_y = row['y_pixel']
            maskData[pixel_y : pixel_y + 5, pixel_x : pixel_x + 5, 0] = 0
            maskData[pixel_y : pixel_y + 5, pixel_x : pixel_x + 5, 1] = 0
            maskData[pixel_y : pixel_y + 5, pixel_x : pixel_x + 5, 2] = 255
            maskData[pixel_y : pixel_y + 5, pixel_x : pixel_x + 5, 3] = 70
        
        mask = QImage(maskData, imageWidth, imageHeight, imageWidth * bytesPerPixel, QImage.Format_ARGB32);
        
        
        painter = QPainter()        
        
        for fileName in imageFileNames:
            fileName = "Data/1999_2017_landsat_time_series/roi/cut/" + fileName[:-1] + ".tif"
            
            raster = gdal.Open(fileName)
            data = Toolbox.readGeotiff(raster)  
            
            # If data is more than one band!
            if len(data.shape) != 3:
                continue      
            if data.shape[0] < 3:
                continue
            x_size = data.shape[2]
            y_size = data.shape[1] 
    
                  
            '''
            ###########
            img = QImage(fileName)
            ptr = img.bits()
            ptr.setsize(img.byteCount())

            ## copy the data out as a string
            strData = ptr.asstring()
            ## get a read-only buffer to access the data
            buf = memoryview(ptr)
            ## view the data as a read-only numpy array            
            arr = np.frombuffer(buf, dtype=np.ubyte).reshape(img.height(), img.width(), 4)
            ## view the data as a writable numpy array
            arr = np.asarray(ptr).reshape(img.height(), img.width(), 4)
            ############            
            '''
            
            # Choose the band:
                    
            ### Bug solved - data type should be declared earlier!
            imageNew = np.zeros((y_size, x_size, 4), dtype=np.uint8) 
            imageNew[:, :, 0] = data[2, :, :]
            imageNew[:, :, 1] = data[1, :, :]
            imageNew[:, :, 2] = data[0, :, :]              
#           if data.shape[0] == 6:
#                imageNew[:, :, 0] = data[2, :, :]
#                imageNew[:, :, 1] = data[1, :, :]
#                imageNew[:, :, 2] = data[0, :, :]
#            else:
#                imageNew[:, :, 0] = data[3, :, :]
#                imageNew[:, :, 1] = data[3, :, :]
#                imageNew[:, :, 2] = data[3, :, :]
            imageNew[:, :, 3] = 255            

            qimg = QImage(imageNew, x_size, y_size, QImage.Format_RGB32)
            
            # put overlay mask
            painter.begin(qimg)
            painter.drawImage(0, 0, mask)
            painter.end()
            
                              
            self.imageLabel.setPixmap(QPixmap.fromImage(qimg)) 
              
            self.fitToWindow()        
            self.updateActions()      
            app.processEvents()
            

            # Change the title
            QMainWindow.setWindowTitle(self, fileName) 
 
            time.sleep(0.2)          
    
    # Show clustered curves
    def kMeansCurve(self):
        fileNameDir = "Data/1999_2017_landsat_time_series/roi/cut/fileNames.txt"
        if 'fileNameDir' in locals():
            pass
        else:
            raise AssertionError("FILE_NAME_DIR is not given!")
            
        with open(fileNameDir, "r") as file:
            imageFileNames = file.readlines()   # imageFiles is a list
            file.close()
        
        dateList = []
        for fileName in imageFileNames:
            # extract date and time from image file names
            year = int(fileName[0:4])
            month = int(fileName[4:6])
            day = int(fileName[6:8])
            dateList.append(dt.date(year, month, day))
        
        pixel_info = pd.read_csv("../pixel_info.csv", sep='\t', encoding='utf-8')
        seriesName = ['Band_1_No._' + str(x+1) for x in range(220)]
        seqArray = pixel_info.as_matrix(columns= [seriesName])
        
        
        ''' Tensorflow clustering '''        
#        n_features = 220
#        n_clusters = 10
#        n_samples_per_cluster = int(len(pixel_info) / n_clusters)
#        seed = 700
#        embiggen_factor = 70        
#        
#        Band_1 = tf.convert_to_tensor(seqArray, np.float32)
#        initial_centroids = choose_random_centroids(Band_1, n_clusters)
#        nearest_indices = assign_to_nearest(Band_1, initial_centroids)
#        updated_centroids = update_centroids(Band_1, nearest_indices, n_clusters)
#        
#        model = tf.global_variables_initializer()
#        with tf.Session() as session:
#            sample_values = session.run(Band_1)
#            updated_centroid_value = session.run(updated_centroids)
#            nearest_indices_value = session.run(nearest_indices)
        
        # Load already clustered results: metadata.tsv
        indicesPd = pd.read_csv('../metadata.tsv', sep='\t', encoding='utf-8')
        indices = indicesPd['0'].astype(int)        
        
        
        # Get clustering indexs
        # First show cluster 1:
        
        # Save figures to one pdf        
        pp = PdfPages('kMeans_Band1_Clusters10.pdf')    
        for no in np.arange(10):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            clusterNo = no
            clusterIndices = (indices == clusterNo)
            clusterSamples = seqArray[clusterIndices]
        
            # randomly choose 100 samples from clusterSamples
            indices10 = []
            for x in np.arange(10):
                indices10.append(np.random.randint(len(clusterSamples)))
            arraySamples10 = clusterSamples[indices10]  
        
           
            #x = np.arange(220)
            for sample in arraySamples10:
                ax.plot(dateList, sample, marker='o', markersize = 3)
                # ax.plot_date(dateList, sample) # plot date dots!
            ax.set_xlim(dateList[0], dateList[-1])
            ax.xaxis.set_major_locator(YearLocator())
            #ax.xaxis.set_major_formatter(DateFormatter('%Y/%m'))
            ax.xaxis.set_major_formatter(DateFormatter('%Y'))
            ax.set_ylim(0, 255)
            fig.autofmt_xdate()            
            plt.title('Cluster %i' %(no+1))
            plt.xlabel('Year', fontsize = 14)
            plt.ylabel('Pixel Intensity')
            plt.show()             
            
            pp.savefig(fig)
        pp.close()
        
#        # Show image sequence
#        fileNameDir = "Data/1999_2017_landsat_time_series/roi/cut/fileNames.txt"
#        if 'fileNameDir' in locals():
#            pass
#        else:
#            raise AssertionError("FILE_NAME_DIR is not given!")
#            
#        with open(fileNameDir, "r") as file:
#            imageFileNames = file.readlines()   # imageFiles is a list
#            file.close()   
#            
#        
#        NDVI_seq = []
#        fig, ax = plt.subplots()
#        
#        for fileName in imageFileNames:
#            fileNameNew = "Data/1999_2017_landsat_time_series/roi/cut/" + fileName[:-1] + ".tif"
#            
#            raster = gdal.Open(fileNameNew)
#            data = Toolbox.readGeotiff(raster)  
#            # close dataset
#            raster = None
#            
#            # If data is more than one band!
#            if len(data.shape) != 3:
#                continue      
#            if data.shape[0] < 3:
#                continue
#            x_size = data.shape[2]
#            y_size = data.shape[1] 
#
#    
#                  
#            '''
#            ###########
#            img = QImage(fileName)
#            ptr = img.bits()
#            ptr.setsize(img.byteCount())
#
#            ## copy the data out as a string
#            strData = ptr.asstring()
#            ## get a read-only buffer to access the data
#            buf = memoryview(ptr)
#            ## view the data as a read-only numpy array            
#            arr = np.frombuffer(buf, dtype=np.ubyte).reshape(img.height(), img.width(), 4)
#            ## view the data as a writable numpy array
#            arr = np.asarray(ptr).reshape(img.height(), img.width(), 4)
#            ############            
#            '''
#            
#            # Choose the band:
#                    
#            ### Bug solved - data type should be declared earlier!
#            imageNew = np.zeros((y_size, x_size, 4), dtype=np.uint8) 
#            imageNew[:, :, 0] = data[2, :, :]
#            imageNew[:, :, 1] = data[1, :, :]
#            imageNew[:, :, 2] = data[0, :, :]              
##           if data.shape[0] == 6:
##                imageNew[:, :, 0] = data[2, :, :]
##                imageNew[:, :, 1] = data[1, :, :]
##                imageNew[:, :, 2] = data[0, :, :]
##            else:
##                imageNew[:, :, 0] = data[3, :, :]
##                imageNew[:, :, 1] = data[3, :, :]
##                imageNew[:, :, 2] = data[3, :, :]
#            imageNew[:, :, 3] = 255            
#
#            qimg = QImage(imageNew, x_size, y_size, QImage.Format_RGB32)
#            
#            self.imageLabel.setPixmap(QPixmap.fromImage(qimg))        
#              
#            self.fitToWindow()
#            self.scaleFactor = 2.0
#            self.updateActions()      
#            app.processEvents()
#            
#
#            # Change the title
#            QMainWindow.setWindowTitle(self, fileName)            
#           
#
#            self.imageLabel.setObjectName("image")
#            self.imageLabel.mousePressEvent = self.getPos
#            
#            actualX = self.x * self.scaleFactor
#            actualY = self.y * self.scaleFactor
##            qcolor = QColor(qimg.pixel(actualX, actualY))
##            # BGR
##            red = qcolor.red()
##            green = qcolor.green()
##            blue = qcolor.blue()
##
##            print("Scalefactor is: ", self.scaleFactor)
#            print("Cursor locations x & y:", actualX, actualY)
##            print("Red, green, blue values: ", red, green, blue)
#            
#            actualX = int(actualX)
#            actualY = int(actualY)
#            
#            if data.shape[0] >= 4:
#                NDVI = Toolbox.NDVI_3by3(data[:, actualY, actualX])
#                NDVI_seq.append(NDVI)     
#            
#            ax.cla()
#            plt.plot(NDVI_seq)
#            ax.set_title(fileName)
#            # plt.pause(0.1)            
#            
#            time.sleep(0.02)    
#            
#            del data, imageNew, qimg
        

    def paintEvent(self, event=None):
        painter = QPainter(self)

        painter.setOpacity(0.7)
        painter.setBrush(Qt.white)
        painter.setPen(QPen(Qt.white))   
        painter.drawRect(self.rect())    
    
    def createActions(self):
        self.openAct = QAction("&Open...", self, shortcut="Ctrl+O",
                triggered=self.open)

        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q",
                triggered=self.close)

        self.zoomInAct = QAction("Zoom &In (25%)", self, shortcut="Ctrl++",
                enabled=False, triggered=self.zoomIn)

        self.zoomOutAct = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-",
                enabled=False, triggered=self.zoomOut)

        self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+S",
                enabled=False, triggered=self.normalSize)

        self.fitToWindowAct = QAction("&Fit to Window", self, enabled=False,
                checkable=True, shortcut="Ctrl+F", triggered=self.fitToWindow)

        self.aboutAct = QAction("&About", self, triggered=self.about)

        self.aboutQtAct = QAction("About &Qt", self,
                triggered=QApplication.instance().aboutQt)
        
        self.viewAct = QAction("View Time Series", self,
                               triggered=self.view)
        
        self.bandViewAct = QAction("View 3 Bands", self,
                                   triggered=self.bandView)
        
        self.NDVIViewAct = QAction("View NDVI..", self,
                                   triggered=self.viewNDVI)
        
        self.kMeansOverlayAct = QAction("kMeans Overlay", self,
                                 triggered=self.kMeansOverlay)
        self.kMeansCurveAct = QAction("kMeans Curve", self,
                                triggered=self.kMeansCurve)

    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.viewMenu = QMenu("&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        #self.helpMenu = QMenu("&Help", self)
        #self.helpMenu.addAction(self.aboutAct)
        #self.helpMenu.addAction(self.aboutQtAct)
        
        self.TimeSeriesMenu = QMenu("&TimeSeries", self)
        self.TimeSeriesMenu.addAction(self.viewAct)
        self.TimeSeriesMenu.addAction(self.bandViewAct)
        self.TimeSeriesMenu.addAction(self.NDVIViewAct)
        
        self.ClusteringMenu = QMenu("&Clustering", self)
        self.ClusteringMenu.addAction(self.kMeansOverlayAct)
        self.ClusteringMenu.addAction(self.kMeansCurveAct)
        

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        #self.menuBar().addMenu(self.helpMenu)        
        self.menuBar().addMenu(self.TimeSeriesMenu)
        self.menuBar().addMenu(self.ClusteringMenu)

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                                + ((factor - 1) * scrollBar.pageStep()/2)))


if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    imageViewer = ImageViewer()
    imageViewer.show()
    sys.exit(app.exec_())

