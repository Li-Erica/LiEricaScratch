__author__ = 'naterussell'
"""
Written by Nathan Russell, ntrusse2@illinois.edu 12/12/2015

Script Description:
The purpose of this script is to offer a quick and dirty solution
as requested by Chris Xu & Sean Xu of Motorola Solutions

Programmer's Notes:
1. Keys used for similarity matching do not form a unique super key for the output columns
This results in ties. The current selection process is arbitrary for ties.
2. It is still not clear to me how the two tables should be merged
3. The similarity metric is currently an unweighted sum of edit distances.
Picking good weights will result in better results

To Reproduce results:
Simply adjust the paths in the test() function and run the python script
Command line options can be added with argparse package one code is stable

Dependencies: (all can be installed from PYPI using pip)
Python 2.7, Numpy, Pandas, multiprocessing, itertools
editdistance, time
"""

import multiprocessing as mp
import itertools
import pandas as pd
import numpy as np
import editdistance as ed
from time import time

def similarity(A,B,W=None):
    """
    Compute Similarity between two sets of strings A and B

    :param A: List of strings
    :param B: List of strings
    :return: numeric score. wethed string similarity
    """
    l_vec = [(ed.eval(a,b)) for a,b in zip(A,B)]

    return np.sum(l_vec)

def concat(*args):
    strs = [str(arg) for arg in args if not pd.isnull(arg)]
    return '/t'.join(strs) if strs else np.nan
np_concat = np.vectorize(concat)

def recursive_concat(df,cols):
    """
    :return:
    """
    r,c = df.shape
    x = ['']*r
    for c in cols:
        x = np_concat(x,df[c])
    return x

def moto_merge(RR_df,MSI_df,P=2):
    """
    Performs a merge of the RR and MSI data sets. Merges based on the best available match

    :param RR_df: pandas Dataframe of RR data
    :param MSI_df: pandas Dataframe of MSI data
    :param P: The number of processes to decompose the problem into. Since these are not threads
              it is possible to set P above the number of threads on the chip and get a speed boost
              due to over-decomposition
    :return:
    """

    # Dictionary contains comparison key pairings
    comparison_mapping = {"County":"COUNTY NAME ULT",
                         "City":"CITY ULT",
                         "RR System Name":"CUSTOMER NAME",
                         "description":"CUSTOMER NAME",
                         "group":"CUSTOMER NAME",
                         "tag":"CUSTOMER NAME",
                         "RR System Name":"LOCATION CUSTOMER NAME ULT",
                         "description":"LOCATION CUSTOMER NAME ULT",
                         "group":"LOCATION CUSTOMER NAME ULT",
                         "tag":"LOCATION CUSTOMER NAME ULT"}

    RR_Output_Cols = [ "RR System Name","State","County","City",
                       "description","tag","group"]
    MSI_Output_Cols = ["COUNTY NAME ULT","CITY ULT","CUSTOMER NAME",
                       "LOCATION CUSTOMER NAME ULT","CUSTOMER NUM",
                       "STATE NAME ULT"]

    RR_keys = np.unique(comparison_mapping.keys())
    MSI_keys = np.unique([comparison_mapping.get(k) for k in comparison_mapping.keys()])

    # Convert to string type
    print("\nType Conversion...")
    t0 = time()
    for k in RR_keys:
        RR_df[k] = RR_df[k].astype(str)
    for k in MSI_keys:
        MSI_df[k] = MSI_df[k].astype(str)
    print("Time (s): "+str(time()-t0))


    rr_rows, rr_cols = RR_df.shape
    msi_rows, msi_cols = MSI_df.shape

    print("\nForm MetaKeys in original table...")
    t0 = time()
    RR_df["RR_MetaKey"] = recursive_concat(RR_df,comparison_mapping.keys())
    MSI_df["MSI_MetaKey"] = recursive_concat(MSI_df,[comparison_mapping.get(k) for k in comparison_mapping.keys()])
    rr_meta = np.unique(RR_df['RR_MetaKey'])
    msi_meta = np.unique(MSI_df['MSI_MetaKey'])
    print("Time (s): "+str(time()-t0))


    print("RR Original Unique: "+str(rr_rows))
    print("RR Metakey Unique: "+str(len(rr_meta)))
    print("RR Output Unique: "+str(len(np.unique(recursive_concat(RR_df,RR_Output_Cols)))))
    print("MSI Original Unique: "+str(msi_rows))
    print("MSI Metakey Unique: "+str(len(msi_meta)))
    print("MSI Output Unique: "+str(len(np.unique(recursive_concat(MSI_df,MSI_Output_Cols)))))


    #print("\n!!!!Warning!!!!!\nShrinking problem size for testing purposes")
    #rr_meta  = rr_meta[0:2000]


    # Serial Matching
    if P == 1:
        print("\nUnique MetaKey Serial Matching...")

        sim_array = np.zeros(len(rr_meta))
        rrmsi_temp = 'NA'
        rr_msi = ['NA']*len(rr_meta)

        #  Iterate over left table RR
        t0 = time()
        for r,c in zip(rr_meta,range(len(rr_meta))):
            # Initialize values for new RR metakey
            best = 1000
            r_vec = r.split('\t')

            # Check through MSI candidate pairs
            for m in msi_meta:
                m_vec = m.split('\t')
                d = similarity(r_vec,m_vec)

                # Check for improvement and update
                # updating temp single variables since..
                # ..easier to update than list and array values
                if d < best:
                    best = d
                    rrmsi_temp = m

            # time estimate loop
            if c == 1000:
                t1k = time()
                est = (((t1k-t0)/1000)*(len(rr_meta)-1000))/(60)
                print("Estimated Time to Completion (min): "+str(est))

            # Update the list and array with current value in temp single variables
            sim_array[c] = best
            rr_msi[c] = rrmsi_temp

        print("Time (min): "+str((time()-t0)/60))

    # Parallel Matching
    else:
        print("\nUnique MetaKey Parallel Process Matching...")
        print("# of processes = "+str(P))
        t0 = time()

        # Define an output queue
        sim_output = mp.Queue()
        rrmsi_output = mp.Queue()

        # Breakup work
        rr_meta_chunked = list(chunks(rr_meta,(P)))
        print(len(rr_meta_chunked))

        # Setup a list of processes that we want to run
        processes = [mp.Process(target=Sim_Match, args=(rr,msi_meta,sim_output,rrmsi_output)) for rr in rr_meta_chunked]

        # Run processes
        for p in processes:
            p.start()

        # Exit the completed processes
        for p in processes:
            p.join()

        # Get process results from the output queue
        sim_results = [sim_output.get() for p in processes]
        rrmsi_results = [rrmsi_output.get() for p in processes]

        # Reformat Results
        sim_array = np.hstack(sim_results)
        rr_msi = list(itertools.chain(*rrmsi_results))
        print("Time (min): "+str((time()-t0)/60))


    # Form Join Df
    print("\nMerging...")
    t0 = time()
    join_df = pd.DataFrame({"RR_MetaKey":rr_meta,
                            "MSI_MetaKey":rr_msi,
                            "Similarity":sim_array
                            })
    # RR to join
    rr_join_df = pd.merge(RR_df, join_df, how='inner', on="RR_MetaKey", left_on=None, right_on=None,
                         left_index=False, right_index=False, sort=False,
                         suffixes=('_x', '_y'), copy=True)

    # RR_join to
    merged_df = pd.merge(rr_join_df, MSI_df, how='inner', on="MSI_MetaKey", left_on=None, right_on=None,
                         left_index=False, right_index=False, sort=False,
                         suffixes=('_x', '_y'), copy=True)

    print("Time (min): "+str((time()-t0)/60))
    r,c = merged_df.shape
    print("\nSim Join\nCol: "+str(c)+" Row: "+str(r))


    return merged_df

def chunks(l, n):
    """Yield successive chunks from l."""
    n = int(np.ceil(len(l)/n))
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def Sim_Match(rr_meta,msi_meta,sim_output,rrmsi_output):
    """
    Inner loop of Sim Merge
    """

    sim_array = np.zeros(len(rr_meta))
    rrmsi_temp = 'NA'
    rr_msi = ['NA']*len(rr_meta)

    #  Iterate over left table RR
    t0 = time()
    for r,c in zip(rr_meta,range(len(rr_meta))):
        # Initialize values for new RR metakey
        best = 1000
        r_vec = r.split('\t')

        # Check through MSI candidate pairs
        for m in msi_meta:
            m_vec = m.split('\t')
            d = similarity(r_vec,m_vec)

            # Check for improvement and update
            # updating temp single variables since..
            # ..easier to update than list and array values
            if d < best:
                best = d
                rrmsi_temp = m

        # Update the list and array with current value in temp single variables
        sim_array[c] = best
        rr_msi[c] = rrmsi_temp

    sim_output.put(sim_array)
    rrmsi_output.put(rr_msi)

def read_moto_csv(RR_path,MSI_path):
    """
    Reads flat text files from directory
    :return:
    """
    print("\nReading "+str(RR_path)+" into memory...")
    t0 = time()
    rr_df = pd.read_csv(RR_path)
    print("Read complete\nTime used (s): "+str(time()-t0))
    r,c = rr_df.shape
    print("Col: "+str(c)+" Row: "+str(r))
    print("Data Types and Columns:")
    print(rr_df.columns.to_series().groupby(rr_df.dtypes).groups)

    print("\nReading "+str(MSI_path)+" into memory...")
    t0 = time()
    msi_df = pd.read_csv(MSI_path)
    print("Read complete\nTime used (s): "+str(time()-t0))
    r,c = msi_df.shape
    print("Col: "+str(c)+" Row: "+str(r))
    print("Data Types and Columns:")
    print(msi_df.columns.to_series().groupby(msi_df.dtypes).groups)

    return (rr_df, msi_df)

def test():
    """
    Simple test script designed to test functionality
    on developers laptop. Not a comprehensive test by any means
    :return:
    """

    # Linux testing
    #rr = "/home/nate/Desktop/Moto_Merge/RR Trunked System crawler data set.csv"
    #msi = "/home/nate/Desktop/Moto_Merge/CA Strategy extract SAMPLE 121115.csv"

    # Windows Testing
    rr = "C:\\Users\\nate\\Desktop\\Moto_Merge\\RR Trunked System crawler data set.csv"
    msi = "C:\\Users\\nate\\Desktop\\Moto_Merge\\CA Strategy extract SAMPLE 121115.csv"

    # Output Directory
    outpath = "C:\\Users\\nate\\Desktop\\Moto_Merge\\merged.csv"


    line = '-'*100+'\n'
    print("\nReading"+line)
    rr_df,msi_df = read_moto_csv(rr,msi)
    print("\nMerge"+line)
    ans = moto_merge(rr_df,msi_df)
    ans.to_csv(outpath)

if __name__ == "__main__":
    test()


