# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 03:46:24 2021

@author: CORE i7
"""

#startimplementation
import pandas as pd
import math,os,sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sqlite3

from sklearn import tree,svm,neighbors
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.linear_model import SGDClassifier, LogisticRegression, LinearRegression , BayesianRidge ,Ridge,LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.base import BaseEstimator
#from lmfit.models import LorentzianModel
import pickle
from numpy import arange
import re
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


        
class createfr:

    # This class enables us to create a table in an sql3 lite database, 
   


    ''' We create a database and make connection to the database''' 
    
    conn = sqlite3.connect('qfx_3.db')        
    c = conn.cursor()
    engine = create_engine('sqlite:///qfx_3.db')
    
    def __init__(self, a,b,tblname):
        
        ''' 
        The variables of this class include, 
        1. The number of colums of our table in a range (a,b)  where a =1 always, 
        2. b is the last column number 
        3.tblname, is the name of our table that we want to create and is a string literal, e.g 'Ideal 
        
        '''

        self.a = a
        self.b =b
        self.tblname = tblname

        
    
    def idgener(self,a,b):
        
        '''
        generate a list of column names for our Ideal and training tables
        return: list of column names of our desired table 
        
        '''
        try:
            self.seq_y = []
            self.seq_y.append('x float')
            for i in range(a,b):
                
                x= 'y'+str(i) + ' float'
                self.seq_y.append(x)
            return  self.seq_y  
        
        except Exception as e:
            
            '''
             catches exceptions
            
            '''
            print(' Error on line {}' .format(sys.exc_info()[-1].tb_lineno),type(e).__name__,e)
    
            
            
    
    def idgtest(self,a,b):
        
      '''
      generate a x and y column names for our test table
      return: list of column names of our desired table 
        
      '''
      try:
          self.seq_y = []
          self.seq_y.append('x float')
                        
          x= 'y float'
          self.seq_y.append(x)
          return self.seq_y
      
      except Exception as e:
          print(' Error on line {}' .format(sys.exc_info()[-1].tb_lineno),type(e).__name__,e)
        

    
    def crtbl(self,a,b):
        
        ''' 
        
        try :  making connection to  the database and create a new table having columns
        from the list of column names
        
        except : catch exceptions in trying to connect to the data base
        
        '''
        try :
            
            Path('qfx_3.db').touch()
            
            self.conn = sqlite3.connect('qfx_3.db')        
            self.c = self.conn.cursor()
            seq_y = self.idgener(a,b)
            
            
            self.c.execute("CREATE TABLE IF NOT EXISTS tblname (%s)"%",".join(seq_y))
             
            print ("successfully created the "+self.tblname+" table.")  
           
        
        except Exception as e:
            print(' Error on line {}' .format(sys.exc_info()[-1].tb_lineno),type(e).__name__,e)
        

        
        
    def padtf(self,ccsv):
        
        ''' 
        
        create a pandas dataframe from csv and write it to the sq3lite table, 
        variable ccsv would be the string literal name of the csv e.g. 'train'  for the train.csv file
        
        '''
        
        try:
        
            
            '''
            create an ideal dataframe for ideal.csv file
            
            '''
            
            if ccsv == 'ideal' :
                
                try:
                    
                    '''
                    
                    create ideal dataframe and notify that its created
                    
                    '''
                    
                    tblname = pd.read_csv('ideal.csv')
                    
                    print('pandas dataframe for ideal functions created')
                    
              
                except  :
                    
                    '''
                    raise exception found in trying to read ideal csv
                    
                    '''
                    
                    print( ' we encounted an error in reading your Ideal csv file')
                    
                else:
                    
                    ''' 
                    
                    try: write data to sql database and notify if it has successfully done so
                    
                    except: raise all kinds of exceptions that could be found in trying to make the connection to database
                    
                    '''
                
                    try: 
                        
                        tblname.to_sql('ideal', self.conn, if_exists = 'append', index = False)
                        print("Successfully written data to the sql ideal table")
                    
                    except Exception as e:
                        print ('Having trouble with writing data to ideal sql table')
                        print(' Error on line {}' .format(sys.exc_info()[-1].tb_lineno),type(e).__name__,e)
                    
            elif ccsv == 'train':
                
               
                
                try:
                    
 
                    '''
                    
                    create train dataframe and notify that its created
                    
                    '''
 
                    
                    
                    tblname = pd.read_csv('train.csv')
                    print('pandas dataframe for train data created')
                    
                except:


                    '''
                    raise exception found in trying to read train.csv file
                    
                    '''


                    print( ' we encounted an error in reading your Train csv file')
                    
                else:
                    
                    
                    ''' 
                    
                    try: write data to sql train table and notify if it has successfully done so
                    
                    except: raise all kinds of exceptions that could be found in trying to make the connection to database
                    
                    '''
                    
                
                    try: 
                        tblname.to_sql('train', self.conn, if_exists = 'append', index = False)
                        print("Successfully written data to the sql train table")
                    
                    except Exception as e :
                        
                        print(' Error on line {}' .format(sys.exc_info()[-1].tb_lineno),type(e).__name__,e)
  
                    
                        print ('Having trouble with writing data to train sql table')
                      
                    
                     
            elif ccsv == 'test':
                
                
                
                try:
                    
                    '''
                    
                    create test dataframe and notify that its created
                    
                    '''
                    
                    tblname = pd.read_csv('test.csv')
                    print('pandas dataframe for test data created')
            
                except :
                    
                    '''
                    
                    raise exception found in trying to read test.csv file
                    
                    ''' 
                    
                    
                    
                    print( ' we encounted an error in reading your test csv file')
                    
                else:
                    
                    
                    ''' 
                    
                    try: write data to sql test table and notify if it has successfully done so
                    
                    except: raise all kinds of exceptions that could be found in trying to make the connection to database
                    
                    '''
                    
                    
                
                    try: 
                        tblname.to_sql('test', self.conn, if_exists = 'append', index = False)
                        print("Successfully written data to the sql test table")
                    
                    except Exception as e :
                        
                        print ('Having trouble with writing data to test sql table','\n')
                        
                        print(' Error on line {}' .format(sys.exc_info()[-1].tb_lineno),type(e).__name__,e)
  
                      
                   
        except:
            
            
            '''
            
            raise exception if the file selected file does not match any of the three csv files
            
            '''
            print(' There is no match to the csv files to be compiled by this program')
        
        
        else:
            
           
            
            ''' 
            
            Reading data from sql table for interaction
            
            '''
            
            if ccsv == 'ideal':
                '''
                
                try: creating a pandas dataframe that reads/loads data from ideal table
                return: return the created pandas dataframe for use in other program operations
                except: Raise exception when there's problem with loading the sql ideal table'
                
                '''
                
                try:
                    self.idfn = pd.read_sql_table("ideal", con=self.engine)
                    print(" Actively reading from Ideal table ")
                    return self.idfn
                
                
                except Exception as e:
                     
                    print(" Error in reading data from sql Ideal table  ",'\n')
                    
                    print(' Error on line {}' .format(sys.exc_info()[-1].tb_lineno),type(e).__name__,e)
  
                
            elif ccsv == 'test': 
                
                '''
                
                try: creating a pandas dataframe that reads/loads data from test table
                return: return the created pandas dataframe for use in other program operations
                except: Raise exception when there's problem with loading the sql test table'
                
                '''
                
                try:
                    self.tst = pd.read_sql_table("test", con=self.engine)
                    print("Actively reading from test sql table")
                    return self.tst  
                except:
                     
                    print(" Error in reading data from sql Test table  ")
 
                
            elif ccsv == 'train':
                
                '''
                
                try: creating a pandas dataframe that reads/loads data from train table
                return: return the created pandas dataframe for use in other program operations
                except: Raise exception when there's problem with loading the sql train table'
                
                '''
                
        
                try:
                    self.trn = pd.read_sql_table("train", con=self.engine)
                    print("Actively reading the train sql table ")
                    return self.trn
                
                except Exception as e:
                     
                    print(" Error in reading data from sql Ideal table  ")
                    
                    print(' Error on line {}' .format(sys.exc_info()[-1].tb_lineno),type(e).__name__,e)
  
            
           
              
    def selectIDb(self):
        
        ''''
        
        Reads data from ideal table row by row
        
        '''
        self.c.execute("SELECT *FROM ideal")
        
        rows = self.c.fetchall()
        
        for row in rows:
            print(row)
            
    def selectTs(self):
        
        ''''
        
        Reads data from test table row by row
        
        '''
        self.c.execute("SELECT *FROM test")
        
        rows = self.c.fetchall()
        
        for row in rows:
            print(row)
            
    def selectTrn(self):
        
        ''''
        
        Reads data from train table row by row
        
        '''
        self.c.execute("SELECT *FROM train")
        
        rows = self.c.fetchall()
        
        for row in rows:
            print(row)

class SelectBestFit:
    
    ''' 
    
    This class enables us to Select the 4 Function from 50 in the Ideal dataset 
    that best Matches our 4 training sets
    
    
    class inheritance from the createfr class enables us to connect to database and dataframes for
    all 3 datasets train, ideal and test.
        
    '''
        
    pt = createfr(1,51,tblname='ideal') 
    pt.idgener(1,51)
    pt.crtbl(1,51)
    idw = pt.padtf(ccsv='ideal')
    
    
    hts = createfr(1, 2, tblname='test')
    hts.idgtest(1, 2)
    hts.crtbl(1, 2)
    fr = hts.padtf(ccsv='test')
  
    htr = createfr(1, 5, tblname= 'train' )
    htr.idgener(1,5)
    htr.crtbl(1, 5)
    bg = htr.padtf(ccsv='train')
    
    def Ideal_datafr(self):
        
        '''
        
        returns the ideal dataframe instance idw
        
        '''
        return self.idw
    
    def test_df(self):
        
        '''
        
        returns the test dataframe instance fr
        
        '''        
        return self.fr
    
    def train_df(self):
        
        '''
        
        returns the train dataframe instance bg
        
        '''
        return self.bg
    
    
    def Trai_Ide(self): 
       
        '''
        
        Traincolumns creates a list of column names to be used in report creation
        
        try:
            
        creates an empty  csv file  (file.csv) 
        
        iterates through each column of the train dataframe and makes iteration in the ideal dataframe computing mean square
        errors between that train function and 50 ideal functions
        
                
        Only the mean square error of 1.41 is acceptable as the criterion for selection and are written in the csv file creating
        a Report      
        
        The unacceptable are simply printed for observation in console
        
        Open csv file to fin out which fuctions among the 50 are the best fit i.e those with the list mean square error
        
        '''
        
        TrainColumns = ["x","y1","y2","y3","y4"]
        i = -1
        try: 
            
            '''
            
            
            NB: ensure to enter a correct path for the csv file
            
            '''
            
            writer = csv.writer(open("C:/Users/CORE i7/Documents/IUBH/Programming with Python/Assignement/Program1/file.csv", 'w'))

            for train in self.bg:
        
                i=i+1
                
      
                
                for column in self.idw:
                    
                    '''
                    
                    computes the mean square between train each train dataframe and 50 other ideal functions
                    
                    
                    '''
                    try:
                        
                        Mean_Squared_Error = metrics.mean_squared_error(self.bg[train],self.idw[column],
                                                            sample_weight=None,multioutput='uniform_average',squared=True)
                        '''
                        prints the report in console of all the iterations made
                        
                        '''
        
                        print('Train Function',TrainColumns[i], ': against Ideal Function :' , column, "\n\n" )
                        print('Mean Squared Error: ', 
                                          Mean_Squared_Error, "\n\n")
                    
                    except Exception as e:
                        
                        '''
                        
                        Handles the exception when program fails to compute the mean square error
                        
                        '''
                        print(' Hint: Ensure you have not duplicated the columns in the train and ideal tables, othwerise create new database and run again the program')
                        print(' Error on line {}' .format(sys.exc_info()[-1].tb_lineno),type(e).__name__,e)
  


                    else:   
                        
                        '''
                        
                        program continues to report creation
                        
                        '''
                        
                        if Mean_Squared_Error < 1.4142:
                            
                            '''
                            
                            writes to csv creating a report
                            
                            '''
                            writer.writerow(["Mean Square Error",Mean_Squared_Error,'Train Funct',TrainColumns[i], " Chosen Ideal Function",column])
                            
                
                        else:
                            
                            '''
                            
                            prints report of the unacceptable  values
                            
                            '''
                            
                            print("....Ideal Function", column, " is having a mse  of  ",Mean_Squared_Error," and therefore is not Acceptable","\n\n")


        except Exception as e:
            
            '''
            
            Handles the exceptions found when the program failes to write to csv 
            
            correct path for the csv file must be entered
            
            '''
            
            print("HINT: Check that the path and file name for your csv file is correct")
            print(' Error on line {}' .format(sys.exc_info()[-1].tb_lineno),type(e).__name__,e)
  



         
        

class FitModel:
        
    '''
    
    This class defines various helper methods for creating dataframes to be used when fitting models on different functions
    
    Inherites from SelectBestFit class for usage of the dataframe instances 
    
    '''
    
    
    brl = SelectBestFit()
    id_df = brl.Ideal_datafr()
    
    fw = SelectBestFit()
    tr_df = fw.train_df()
    
    n = SelectBestFit()
    ts_df = n.test_df()
    

    
    def idealx(self):
         
        '''
        
        creates a seperate pandas column of the ideal x column 
        
        '''
          
        return   self.id_df.loc[:,['x']]

    def ideal44(self):
         
         
        '''
        
        creates a seperate pandas column of the ideal y44 column 
        
        '''
            
        return    self.id_df.loc[:,['y44']]
    
    def trainx(self):
         
        '''
        
        creates a seperate pandas column of the train x column 
        
        '''
         
        return   self.tr_df.loc[:,['x']] 

    def trainy1(self):
         
         
        '''
        
        creates a seperate pandas column of the train y1 column 
        
        '''
            
        return   self.tr_df.loc[:,['y1']]


    def ideal41(self):
         
         
        '''
        
        creates a seperate pandas column of the ideal y41 column 
        
        '''
            
        return    self.id_df.loc[:,['y41']]
    

    def trainy2(self):
         
         
        '''
        
        creates a seperate pandas column of the train y2 column 
        
        '''
          
        return   self.tr_df.loc[:,['y2']]

    def ideal34(self):
         
         
        '''
        
        creates a seperate pandas column of the ideal y34 column 
        
        '''
          
        return    self.id_df.loc[:,['y34']]
    

    def trainy3(self):
         
                 
        '''
        
        creates a seperate pandas column of the train y3 column 
        
        '''
                  
        return   self.tr_df.loc[:,['y3']]


    def ideal21(self):
         
         
        '''
        
        creates a seperate pandas column of the ideal y21 column 
        
        '''
          
        return    self.id_df.loc[:,['y21']]
    

    def trainy4(self):
         
                 
        '''
        
        creates a seperate pandas column of the train y4 column 
        
        '''
          
        return   self.tr_df.loc[:,['y4']]

    def tstds(self):
         
         
        '''
        
        creates pandas dataframe for the test set
        
        '''
          
        return self.ts_df
        
                     
      
     
class Ppln: 
    
    '''
    
    class fits pipeline models to the selected functions from ideal set and does some plottng for visualization
    
    Inherites dataframes from FitModel and SelectBestFit classes and their helper functions  
    
    '''

    bw = FitModel()
    
    '''
    x ideal function
    
    '''
    ikx =bw.idealx()
    
    '''
    y 21 ideal function
    
    '''
    ik21 = bw.ideal21()
    
    '''
    y 44 ideal function
    
    '''
    ik44 = bw.ideal44()
    
    '''
    y 41 ideal function
    
    '''
    ik41 = bw.ideal41()
    
    '''
    y 34 ideal function
    
    '''
    ik34 = bw.ideal34()
    
    '''
    test data in a pandas dataframe
    
    '''
    iktest = bw.tstds()
    
    tw1 = bw.trainy1()
    tw2 = bw.trainy2()
    tw3 = bw.trainy3()
    tw4 = bw.trainy4()
    twx = bw.trainx()
    
    fw = SelectBestFit()
    tr_df = fw.train_df()
    
    n = SelectBestFit()
    ts_df = n.test_df()                 
    
    
    '''
    
    lasso_eps ,lasso_alpha and lasso_iter are parameters used by the pipeline model to fit data
    
    '''
    lasso_eps = 0.00001
    lasso_nalpha = 10
    lasso_iter=5000
    
    '''
    degree_min, and degree_max parameters defines the range of degreein which our polynomial function must raised
    in this case its from polynomial of degree 1 up to 4
    
    '''
    degree_min = 1
    degree_max = 4

     
    def __init__(self,x_ideal,y_ideal,x_train,y_train):
        
        '''
        
        Class variables for the x,y pair of the ideal and train functions needs to be defined
        
        '''
        
        self.x_ideal = x_ideal
        self.y_ideal = y_ideal
        self.x_train = x_train
        self.y_train = y_train
        
       
    def Pip_lne(self):
        
        '''
        
        1.fits linear polynomial model to the chosen function for mapping new data
        2. plots the model predicted values against its ideal data for comparison and calculates their rmse 
        
        '''
        
        try:
        
            writer = csv.writer(open("C:/Users/CORE i7/Documents/IUBH/Programming with Python/Assignement/Program1/Fittdata2.csv", 'w'))
            
            '''
            iterates through 4 cycles to check which polynomial degree best fits the model function
            fits model to chosen ideal function data
            
            '''
            
            for degree in range(self.degree_min,self.degree_max+1):
                self.model = make_pipeline(PolynomialFeatures(degree, interaction_only=False),LassoCV(eps=self.lasso_eps,
                            n_alphas=self.lasso_nalpha, max_iter=self.lasso_iter, normalize = True, cv=10))
                
                ga = self.model.fit(self.x_ideal, self.y_ideal)
                Y_pred = ga.predict(self.x_ideal)
                rmse = np.sqrt(mean_squared_error(self.y_ideal, Y_pred))
                
                try: 
                    '''
                
                    writes a csv report recording every polynomial degree and their rmse
                    
                    
                    '''
                    
                    writer.writerow([rmse, '(rmse)', degree, 'Polynomial Degree','yideal',self.y_ideal])
                   
                    '''
                    scatter plots  of train data against the chosen ideal function
                    
                    '''
                    plt.scatter(self.x_train, self.y_train, color ='red')
                    
                    '''
                    plot the chosen ideal x values against the predicted values
                    
                    '''
                    plt.plot(self.x_ideal,ga.predict(self.x_ideal), color='blue')
                    
                    
                    '''
                    x label 
                    
                    '''
                    
                    plt.xlabel('x test data')
                    
                
                    
                    '''
                    y label
                    
                    ''' 
                    
                    plt.ylabel('y test data')
                    
                    '''
                    show the superimposed plots
                    
                    '''
                    try:
                        '''
                        title of plot
                        
                        '''
                        if self.y_ideal == self.ik21:
                            
                            '''
                            
                            title plot for y4 train 
                            
                            '''
                        
                            plt.title('y4 training vs chosen y21 ideal function : Prediction ')
                        elif self.y_ideal == self.ik44:
                            
                            '''
                            
                            title plot for y1 train 
                            
                            '''
                        
                            plt.title(' y1 training vs chosen y44 ideal function : Prediction ')
                         
                        elif self.y_ideal == self.ik41:
                            
                            '''
                            
                            title plot for y2 train 
                            
                            '''
                        
                            plt.title('y2 training vs chosen y41 ideal function : Prediction ')
                        elif self.y_ideal == self.ik34:
                            
                            '''
                            
                            title plot for y3 train 
                            
                            '''
                        
                            plt.title('y3 training vs chosen y34 ideal function : Prediction ')
                            
                    except Exception as e:
                        print(' Error on line {}' .format(sys.exc_info()[-1].tb_lineno),type(e).__name__,e)
                
                except Exception as e: 
                    
                            print(' Error on line {}' .format(sys.exc_info()[-1].tb_lineno),type(e).__name__,e)
                   
            
            
                else:
                 plt.show()
                    
                    
        
        except Exception as e:
            
            '''
            
            handles any  exception including that when the program fails to write data to csv or to fit the model 
            
            '''
            
            print("Hint : Ensure the path to csv is correct ")
            print(' Error on line {}' .format(sys.exc_info()[-1].tb_lineno),type(e).__name__,e)
            
       
            
            
     
    def plotes(self):
        
        '''
        
        fits the model to the chosen function  and plots it, this time using a specific polynomial degree and not a range
        Pip_lne has guided us to know which degree is accurate for each function
        
        '''
        
        
        hlow =1
        hhigh =4
        
        for draw in range(hlow,hhigh):
            
            model1 = make_pipeline(PolynomialFeatures(1, interaction_only=False),LassoCV(eps=self.lasso_eps,
                  n_alphas=self.lasso_nalpha, max_iter=self.lasso_iter, normalize = True, cv=10))
               
            model2 = make_pipeline(PolynomialFeatures(3, interaction_only=False),LassoCV(eps=self.lasso_eps,
                   n_alphas=self.lasso_nalpha, max_iter=self.lasso_iter, normalize = True, cv=10))
 
            if draw ==1:
                
                '''
                
                fits ideal y21 to a polynomial degree of order 3
                make scatter plot of x training against y4 training data in red, to compare with the predicted plot in blue
                
                '''
        
                p=model2.fit(self.ikx,self.ik21)
                plt.scatter(self.twx, self.tw4, color ='red')
                
                '''
                maps new predicted values using model fit and  plots against x ideal
                
                '''
                
                plt.plot(self.ikx,p.predict(self.ikx), color='blue')
                
                '''
                plot titled as y4 training against y21 ideal function prediction model
                
                '''
                plt.title('y4 training vs chosen y21 ideal function Prediction model ')
                
                '''
                
                 x label of plot labelled as x train
                
                '''
                plt.xlabel('x train data')
                
                '''
                
                y label of plot as y4 train data
                
                '''
                plt.ylabel('y4 train data')
                
                '''
                
                displays the plots made
                
                '''
                
                plt.show()      
                
            elif draw ==2 :
                
                '''
                
                fits ideal y44 to a polynomial degree of order 1
                make scatter plot of x training against y1 training data in red, to compare with the predicted plot in blue
                
                '''
          
                model1.fit(self.ikx,self.ik44)
                plt.scatter(self.twx, self.tw1, color ='red')
                
                '''
                maps new predicted values using model fit and  plots against x ideal
                
                '''
                                
                plt.plot(self.ikx,model1.predict(self.ikx), color='blue')
                
                '''
                plot titled as y1 training against y44 ideal function prediction model
                
                '''
                plt.title('y1 training vs chosen y44 ideal function prediction model')
                
                '''
                
                 x label of plot labelled as x train
                
                '''
                
                plt.xlabel('x train data')
                
                '''
                
                y label of plot as y1 train data
                
                '''
                
                plt.ylabel('y1 train data')
                
                '''
                
                displays the plots made
                
                '''
                
                plt.show()      
 
            elif draw ==3 :
                
                '''
                
                fits ideal y41 to a polynomial degree of order 3
                make scatter plot of x training against y2 training data in red, to compare with the predicted plot in blue
                
                '''
          
                model2.fit(self.ikx,self.ik41)
                plt.scatter(self.twx, self.tw2, color ='red')
                
                '''
                maps new predicted values using model fit and  plots against x ideal
                
                '''
                   
                plt.plot(self.ikx,model2.predict(self.ikx), color='blue')
                
                '''
                plot titled as y2 training against y41 ideal function prediction model
                
                '''
                plt.title('y2 training vs chosen y41 ideal function prediction model ')
                
                '''
                
                 x label of plot labelled as x train
                
                '''
                
                plt.xlabel('x train data')
                
                '''
                
                y label of plot as y2 train data
                
                '''
                 
                plt.ylabel('y2 train data')
                
                '''
                
                displays the plots made
                
                '''
                
                plt.show()   
                
            elif draw ==4 :
                
                '''
                
                fits ideal y34 to a polynomial degree of order 3
                make scatter plot of x training against y3 training data in red, to compare with the predicted plot in blue
                
                '''
          
                model2.fit(self.ikx,self.ik34)
                plt.scatter(self.twx, self.tw3, color ='red')
                
                '''
                maps new predicted values using model fit and  plots against x ideal
                
                '''
                
                plt.plot(self.ikx,model2.predict(self.ikx), color='blue')
                
                '''
                plot titled as y3 training against y34 ideal function prediction model
                
                '''
                plt.title('y3 training vs chosen y34 ideal function prediction model ')
                
                '''
                
                 x label of plot labelled as x train
                
                '''
                
                plt.xlabel('x train data')
                
                '''
                
                y label of plot as y3 train data
                
                '''
                 
                plt.ylabel('y3 train data')
                
                '''
                
                displays the plots made
                
                '''
                
                plt.show()      
     
    def chects(self):
        
       '''
        
       1.iterates through test data and selects from the chosen ideal function models the one which maps 
       more accurately on each pair of data
       
       2. for each model a seperate empty list is created and whenever a new pair is found it is appended to that list
       
       3. the appended values are later on changed to a pandas dataframe and subsequently to a database table
       '''
           
       newx1 = []
       newy1 = [] 
       newd1 = []
       newidl =[]
       newd2 = []
       newidl2 =[]
       newd3 = []
       newidl3 =[]
       newd4 = []
       newidl4 =[]
       newx2 = []
       newy2 = []
       newx3 = []
       newy3 = []
       newx4 = []
       newy4 = []
       
       '''
       pipeline polynomial model is constructed for all the 4 ideal functions as done previously
       
       '''
       
       model1 = make_pipeline(PolynomialFeatures(1, interaction_only=False),LassoCV(eps=self.lasso_eps,
           n_alphas=self.lasso_nalpha, max_iter=self.lasso_iter, normalize = True, cv=10))
       
       model2 = make_pipeline(PolynomialFeatures(3, interaction_only=False),LassoCV(eps=self.lasso_eps,
           n_alphas=self.lasso_nalpha, max_iter=self.lasso_iter, normalize = True, cv=10))
       
       model3 = make_pipeline(PolynomialFeatures(3, interaction_only=False),LassoCV(eps=self.lasso_eps,
           n_alphas=self.lasso_nalpha, max_iter=self.lasso_iter, normalize = True, cv=10))
       
       model4 = make_pipeline(PolynomialFeatures(3, interaction_only=False),LassoCV(eps=self.lasso_eps,
           n_alphas=self.lasso_nalpha, max_iter=self.lasso_iter, normalize = True, cv=10))
       
       '''
       
       the model is fit into our data
       
       '''       
       
       m1= model1.fit(self.ikx,self.ik44)
       m2= model2.fit(self.ikx,self.ik41)
       m3= model3.fit(self.ikx,self.ik34)       
       m4= model4.fit(self.ikx,self.ik21)
       
       '''
       empty pandas dataframe are created to later hold our selected test pairs of data
       
       '''
       
       f1 = pd.DataFrame()
       f2 = pd.DataFrame()
       f3 = pd.DataFrame()
       f4 = pd.DataFrame()   
       
       '''
       iterates through the test rows and cross checks which model best fits the data
       
       '''
       
       for index, row in self.iktest.iterrows():
           
           '''
           
           reshaping the data enables us to use it in calculations
           
           '''
           test = (row['x']).reshape(-1, 1)
           
           '''
           
           subtracts test y value with model predicted value then squares the difference to avoid negative numbers
           this is done for all the 4 different models
           
           '''

           n1 = math.sqrt((row['y']-np.array(m1.predict(test)))**2)
           n2 = math.sqrt((row['y']-np.array(m2.predict(test)))**2)
           n3 = math.sqrt((row['y']-np.array(m3.predict(test)))**2)
           n4 = math.sqrt((row['y']-np.array(m4.predict(test)))**2)
           
           '''
           
           dift selects the minimum value of differences among the 4 obtained
           
           '''
           
           dift= min(n1,n2,n3,n4)
           
           '''
           
           to enable high accuracy in mapping the test values only those with a difference of 2 or less are chosen
           
           '''
           
           if dift <2:
               
               '''
               if the selected pair fits under model 1 then they are appended in a list and then merged to their
               respective dataframe
               
               the chosen ideal function name which is identified with the test data are also recorded accordingly
               
               the value of the difference is given its column as well
               
               '''
               
               if dift == n1 : 
       
                   #Storing the selected values in a new dataframe
                   var1=row.loc['x']
                   var2=row.loc['y']
                   var2x = n1
                   var1x = 'y1'
                   newd1.append(var2x)
                   newidl.append(var1x)
                   newx1.append([var1])
                   newy1.append([var2])
                   f1new = f1.append(pd.DataFrame(newx1))
                   f1new.columns = ['test x']
                   fdel1 = f1.append(pd.DataFrame(newd1))
                   fdel1.columns = ['delta y']   
                   fun1 = f1.append(pd.DataFrame(newidl))
                   fun1.columns = ['function No.']
                   
                   c=f1.append(pd.DataFrame(newy1))
                   c.columns=['test y']
                   gv1 = pd.merge(f1new,c, right_index=True, left_index=True)
                   gv1x = pd.merge(fdel1,fun1, right_index=True, left_index=True)
                   gv1g = pd.merge(gv1,gv1x,right_index=True, left_index=True)
                   c['y']= f1new
                   print(index, ' ' ,dift,'funct 1')
                       
       

               elif dift == n2:
                   
                    var3=row.loc['x']
                    var4=row.loc['y']
                    var4x = n2
                    var3x = 'y2'
                    newd2.append(var4x)
                    newidl2.append(var3x)
                    newx2.append([var3])
                    newy2.append([var4])
                    f2new = f2.append(pd.DataFrame(newx2))
                    f2new.columns = ['test x']
                    
                    fdel2 = f1.append(pd.DataFrame(newd2))
                    fdel2.columns = ['delta y']   
                    fun2 = f1.append(pd.DataFrame(newidl2))
                    fun2.columns = ['function No.']
                    
                    d=f2.append(pd.DataFrame(newy2))
                    d.columns=['test y']
                    gv2 = pd.merge(f2new,d, right_index=True, left_index=True)
                    
                    gv2x = pd.merge(fdel2,fun2, right_index=True, left_index=True)
                    gv2g = pd.merge(gv2,gv2x,right_index=True, left_index=True)

                    d['y']= f2new
                      
                   
                    print(index, ' ' ,dift,'funct 2')
                       
               elif dift ==n3:
                   
                    var5=row.loc['x']
                    var6=row.loc['y']
                    var6x = n3
                    var5x = 'y3'
                    newd3.append(var6x)
                    newidl3.append(var5x)
                    newx3.append([var5])
                    newy3.append([var6])
                    f3new = f3.append(pd.DataFrame(newx3))
                    f3new.columns = ['test x']
                    
                    fdel3 = f1.append(pd.DataFrame(newd3))
                    fdel3.columns = ['delta y']   
                    fun3 = f1.append(pd.DataFrame(newidl3))
                    fun3.columns = ['function No.']
                   
                    e=f3.append(pd.DataFrame(newy3))
                    e.columns=['test y']
                    gv3 = pd.merge(f3new,e, right_index=True, left_index=True)

                    gv3x = pd.merge(fdel3,fun3, right_index=True, left_index=True)
                    gv3g = pd.merge(gv3,gv3x,right_index=True, left_index=True)

                    e['y']= f3new
  
                   
                    print(index, ' ' ,dift,'funct 3')
                       
               elif dift ==n4:
                   
                    var7=row.loc['x']
                    var8=row.loc['y']
                    var8x = n4
                    var7x = 'y4'
                    newd4.append(var8x)
                    newidl4.append(var7x)
                    newx4.append([var7])
                    newy4.append([var8])
                    f4new = f4.append(pd.DataFrame(newx4))
                    f4new.columns = ['test x']
                    
                    fdel4 = f1.append(pd.DataFrame(newd4))
                    fdel4.columns = ['delta y']   
                    fun4 = f1.append(pd.DataFrame(newidl4))
                    fun4.columns = ['function No.']
                    
                    
                    f=f4.append(pd.DataFrame(newy4))
                    f.columns=['test y']
                    gv4 = pd.merge(f4new,f, right_index=True, left_index=True)

                    gv4x = pd.merge(fdel4,fun4, right_index=True, left_index=True)
                    gv4g = pd.merge(gv4,gv4x,right_index=True, left_index=True)

                    f['y']= f4new
  
                   
                    print(index, ' ' ,dift,'funct 4')
                      
       
       
       '''
       
       merge all the different selected test data pairs into a single dataframe
       
       '''
       
       allGVF = [gv1g,gv2g,gv3g,gv4g]
       vt = createfr(1, 51, 'Devi')
       vt.c
       GVF= pd.concat(allGVF).reset_index(drop=True)
       
       
       '''
       try: create a table Dev_ma into sq3 lite database and write data from our merged dataframe
       except: handles all different types of exceptions that are obtained in trying to create a table in our sq3lite
       database
       '''
       try:
           
           sqlite_table = "Dev_ma"
           GVF.to_sql(sqlite_table,vt.engine,if_exists= 'fail')                 
           print(GVF)
           
           
       except Exception as e:
           print(' Error on line {}' .format(sys.exc_info()[-1].tb_lineno),type(e).__name__,e)

       

   
                

def main():
    
    '''
    create ideal table and populate it with data 
    
    '''
    ph = createfr(1,51,tblname='ideal')
    ph.idgener(1,51)
    ph.crtbl(1,51)
    ph.padtf(ccsv='ideal')
   
    '''
    create test table and populate it with data 
    
    '''
    
    htest = createfr(1, 2, tblname='test')
    htest.idgtest(1, 2)
    htest.crtbl(1, 2)
    htest.padtf(ccsv='test')
    
    '''
    create train table and populate it with data 
    
    '''
    
    htrain = createfr(1, 5, tblname= 'train' )
    htrain.idgener(1,5)
    htrain.crtbl(1, 5)
    htrain.padtf(ccsv='train')
    
    '''
    Select from only 4 best fit functions from the given 50 ideal functions
    csv file is created - check to see which functions are the best fit  
    
    '''
    
    bf = SelectBestFit()
    bf.Trai_Ide()
    
    
    '''
    create x,y pairs of data from chosen ideal functions and training data 
    
    '''
    cs = FitModel()
    y44 = cs.ideal44()
    xidl = cs.idealx()
    xtrs = cs.trainx()
    ytr1 = cs.trainy1()
    ytr2 = cs.trainy2()
    ytr3 = cs.trainy3()
    ytr4 = cs.trainy4()
    y21 = cs.ideal21()
    y34 = cs.ideal34()
    y41 = cs.ideal41()
      
    
    '''
    class instance for Ppln to fit y44 ideal for making some analysis
    
    '''
    y44cg = Ppln(xidl, y44, xtrs, ytr1)
    
    '''
    y44cg.Pip_lne(): fit y44 ideal to compare with y1 training making some plots
    order of polynomial unknown
    
    '''  
    
    
    y44cg.Pip_lne()
    
    '''
    y44cg.plotes(): fit y44 ideal to compare with y1 training making some plots
    order of polynomial known
    
    '''    
    
    y44cg.plotes()
      
    
    '''
    class instance for Ppln to fit y41 ideal for making some analysis
    
    '''
    y41cg = Ppln(xidl, y41, xtrs, ytr2)
    
    '''
    y41cg.Pip_lne(): fits y41 ideal to the compare with y2 training making some plots
    order of polynomial unknown
    
    '''  
        
    y41cg.Pip_lne()
    
        
    
    '''
    class instance for Ppln to fit y34 ideal for making some analysis
    
    '''
    y34cg = Ppln(xidl, y34, xtrs, ytr3)
    
    '''
    y34cg.Pip_lne(): fits y21 ideal to the compare with y4 training making some plots
    order of polynomial unknown
    
    '''  
        
    y34cg.Pip_lne()
    
    '''
    class instance for Ppln to fit y21 ideal for making some analysis
    
    '''
    y21cg = Ppln(xidl, y21, xtrs, ytr4)
    
    '''
    y21.Pip_lne(): fits y21 ideal to the compare with y4 training making some plots
    order of polynomial unknown
    
    '''  
       
    y21cg.Pip_lne()
    
        
    '''
    selects test data and the function model which can best map them and finally creating a database table
    showing their differences
    
    '''    
    y44cg.chects()
  
    

if __name__ == "__main__":
    main()
        
