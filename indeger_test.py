# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 17:56:00 2021

@author: CORE i7
"""
from PrograAsgm import createfr,SelectBestFit,FitModel
import unittest
import pandas as pd


'''

Hint Before running the programme create a new database in the module package to avoid overwritting of data

'''
class TestIngen(unittest.TestCase):
    
    '''
    
    tests if the helper methods in createfr() class are working as expected
    
    '''
    
    def test_idgener(self):
        
        '''
        tests if ideal and train table column's elements are accurate
        
        '''
        
        bt = createfr(1,5,'house')
        vf = bt.idgener(1,5)
        self.assertEqual(vf,['x float','y1 float','y2 float','y3 float','y4 float'])
        
    def test_idgtest(self):
        
        '''
        tests if x and y columns for test table are generated accurately
        
        '''
        rt = createfr(1,3,'testd')
        wq = rt.idgtest(1,3)
        self.assertEqual(wq,['x float','y float'])
        
class TestDfr(unittest.TestCase):
    
    '''
    tests if the helper methods in SelectBestFit() class are returning expected dataframe objects
    
    '''
    
    Qa = SelectBestFit()
    Qld= Qa.Ideal_datafr()
    Qts = Qa.test_df()
    Qdr = Qa.train_df()
    Sq = Qa.idw
    St = Qa.fr
    Sd = Qa.bg
    
    def test_Ideal_datafr(self):
        
        '''
        tests if Ideal_datafr is returning the correct Ideal pandas dataframe
        
        '''
                       
        pd.testing.assert_frame_equal(self.Sq, self.Qld)
        
    def test_test_df(self):
        
        '''
        tests if test_df is returning the correct test pandas dataframe
        
        '''
                       
        pd.testing.assert_frame_equal(self.St, self.Qts)   
        
    def test_train_df(self):
        
        '''
        tests if train_df is returning the correct train pandas dataframe
        
        '''
                       
        pd.testing.assert_frame_equal(self.Sd, self.Qdr)  

class Testfit(unittest.TestCase):
    
    '''
    
    tests if the helper methods in FitModels() are returning expected results
    
    
    '''
    
    gr = FitModel()
    
   
                    
    '''
    
    create dataframes from csv
    
    '''
    
    tesdf = pd.read_csv('test.csv')
    idl_df = pd.read_csv('ideal.csv')
    trn_df = pd.read_csv('train.csv')
    
    
    xtrdf = trn_df.loc[:,['x']]
    try1 = trn_df.loc[:,['y1']]
    try2 = trn_df.loc[:,['y2']]
    try3 = trn_df.loc[:,['y3']]
    try4 = trn_df.loc[:,['y4']]
    
    xidf = idl_df.loc[:,['x']]
    id44 = idl_df.loc[:,['y44']]
    id34 = idl_df.loc[:,['y34']]
    id21 = idl_df.loc[:,['y21']]
    id41 = idl_df.loc[:,['y41']]
    
    def test_idealx(self):
        
        '''
        tests if the idealx() method is working excpectedly
        
        '''
        pd.testing.assert_frame_equal(self.xtrdf, self.gr.idealx())
    
      
    def test_ideal44(self):
        
        '''
        tests if the ideal44() method is working excpectedly
        
        '''
        pd.testing.assert_frame_equal(self.id44, self.gr.ideal44()) 
        
    def test_ideal41(self):
        
        '''
        tests if the ideal41() method is working excpectedly
        
        '''
        pd.testing.assert_frame_equal(self.id41, self.gr.ideal41()) 
        
    def test_ideal34(self):
        
        '''
        tests if the ideal34() method is working excpectedly
        
        '''
        pd.testing.assert_frame_equal(self.id34, self.gr.ideal34())
        
        
    def test_ideal21(self):
        
        '''
        tests if the ideal21() method is working excpectedly
        
        '''
        pd.testing.assert_frame_equal(self.id21, self.gr.ideal21()) 
        
    def test_trainx(self):
        
        '''
        tests if the trainx() method is working excpectedly
        
        '''
        pd.testing.assert_frame_equal(self.xtrdf, self.gr.trainx())
        
    def test_trainy1(self):
        
        '''
        tests if the trainy1() method is working excpectedly
        
        '''
        pd.testing.assert_frame_equal(self.try1, self.gr.trainy1())
        
    def test_trainy2(self):
        
        '''
        tests if the trainy2() method is working excpectedly
        
        '''
        pd.testing.assert_frame_equal(self.try2, self.gr.trainy2())
        
    def test_trainy3(self):
        
        '''
        tests if the trainy3() method is working excpectedly
        
        '''
        pd.testing.assert_frame_equal(self.try3, self.gr.trainy3())
        
    def test_trainy4(self):
        
        '''
        tests if the trainy4() method is working excpectedly
        
        '''
        pd.testing.assert_frame_equal(self.try4, self.gr.trainy4())
        
    def test_tstds(self):
        
        '''
        tests if the tstdf() method is working excpectedly
        
        '''
        pd.testing.assert_frame_equal(self.tesdf, self.gr.tstds())
            
        

if __name__ == "__main__":
  
        
  unittest.main()