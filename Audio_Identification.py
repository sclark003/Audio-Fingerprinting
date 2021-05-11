# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 18:35:57 2021

@author: Sarah
"""
import os
import numpy as np
import librosa
import librosa.display
from skimage.feature import peak_local_max
import hashlib
import time
import evaluate
from itertools import groupby

      
# generate hashes for audio fingerp[rint]
def generate_hashes(peaks, max_hash_time_delta = 500, fan_value = 25):

    hashList = []
    for i in range(len(peaks)):
      for j in range(1, fan_value):
        if (i + j) < len(peaks):

          freq1 = peaks[i][0]      # take current & next peak frequency value
          freq2 = peaks[i + j][0]
          t1 = peaks[i][1]         # take current & next -peak time offset
          t2 = peaks[i + j][1]  
          t_delta = t2 - t1        # get diff of time offsets

          # check if delta is below max hash time delta
          if t_delta <= max_hash_time_delta:
              hash_string = str(freq1)+","+str(freq2)+":"+str(t_delta)
              h = hashlib.sha1(hash_string.encode('utf-8'))
              hashList.append((h.hexdigest(), t1))           # add hash and offset to hash list for track

    return hashList

# get constellation points for fingerprint (database) tracks
def fingerPrint(y):
    D = np.abs(librosa.stft(y,n_fft=2048,window='hann',win_length=2048,hop_length=512))  # get time frequency representation
    constellation = peak_local_max(np.log(D),5,0.05,indices = True,num_peaks=5000)       # get peak coordinates
    return constellation


# get constellation points for query track
def getConstellation(y,min_distance=6,threshold_rel=1):
    D = np.abs(librosa.stft(y,n_fft=2048,window='hann',win_length=2048,hop_length=512))                 # get time frequency representation
    constellation = peak_local_max(np.log(D),min_distance,threshold_rel,indices = True,num_peaks=1000)  # get peak coordinates
    return constellation


# find matching hashes
def return_matches(hashes,fingerPrintPath):
        # Create a dictionary of hash => offset pairs for later lookups
        mapper = {}
        for hsh, offset in hashes:
            if hsh.upper() in mapper.keys():
                mapper[hsh.upper()].append(offset)
            else:
                mapper[hsh.upper()] = [offset]

        values = list(mapper.keys())
        results = []
        matches = {}
        count = {}
        
        # search for matches between database tracks and query track
        for entry in os.scandir(fingerPrintPath):
                fingerPrintFile = fingerPrintPath + entry.name
                fingerprint = np.load(fingerPrintFile)              # load database track fingerpront (hashes)
                for hsh,offset in fingerprint:
                    if hsh.upper() in values:
                        if entry.name not in count:
                            count[entry.name] = 1                   # if match is found, add track to count dictionary
                        else:
                            count[entry.name] += 1                  # increase count if more matches are found for chosen track
                        
                        for segment_offset in mapper[hsh.upper()]:  # find time offset between databse track and query track
                            t = int(offset)-segment_offset
                            if t>0:
                                results.append((entry.name,t))
        return results,count   # return 'results = (track name, time offset)', and 'count = number of matching hashes per track'


# align time offsets within hashes
def alignMatches(matches,count):
     # count offset occurrences per song and keep only the maximum ones.
     # song matches finds the number of times an offset has the same time alignment per track
     sorted_matches = sorted(matches, key=lambda m: (m[0], m[1]))  
     counts = [(*key, len(list(group))) for key, group in groupby(sorted_matches, key=lambda m: (m[0], m[1]))]
     songs_matches = sorted([max(list(group), key=lambda g: g[2]) for key, group in groupby(counts, key=lambda count: count[0])],key=lambda count: count[2], reverse=True)

     songs_result = []
     for song_id, offset, _ in songs_matches:  # consider topn elements in the result
            song = song_id
            songs_result.append(song)   # return (identified track, number of alignes hashes)
     
     return songs_result


# build fingerprint database
def fingerprintBuilder(databasePath,fingerprintsPath):
    maxfilestoload = 1000
    i = 0
    for entry in os.scandir(databasePath):
        if i<maxfilestoload:
            filepath = databasePath + entry.name
            fingerpath = fingerprintsPath + entry.name
            y, sr = librosa.load(os.path.join(filepath))  # load audio         
            coordinates = fingerPrint(y)                  # get fingerprint         
            hsh = generate_hashes(coordinates)            # get hashes
            np.save(fingerpath,hsh)                       # save hashes
        i+=1


# identify query track
def audioIdentification(queryPath,fingerPrintPath,output):
    f = open(output,"w")
    maxfilestoload = 20#1000      # max files to load to aid in testing
    i = 0
    results = []
    count_list = []
    for entry in os.scandir(queryPath):
        if i<maxfilestoload:
            queryfile = queryPath + entry.name 
            y, sr = librosa.load(os.path.join(queryfile))                    # load audio
            y = y[220500:441000]
            coordinates = getConstellation(y)                                # get fingerprint            
            hsh = generate_hashes(coordinates)                               # get hashes
            
            matches_count,count = return_matches(hsh,fingerPrintPath)        # get matches
            final_results = alignMatches(matches_count,count)                # identify audio
            results.append((entry.name,final_results))                       # find results for testing
            
            # print results to output file
            if len(final_results)==0:
                  text = entry.name +"  :  Not recognised\n"
            elif len(final_results)==1:
                text = entry.name +"  :  "+ str(final_results[0]) +"\n" 
            elif len(final_results)==2:
                text = entry.name +"  :  "+ str(final_results[0]) +"   "+ str(final_results[1]) +"\n" 
            else:
                text = entry.name +"  :  "+ str(final_results[0]) +"   "+ str(final_results[1]) +"   "+ str(final_results[2]) +"\n" 
            f.write(str(text))
        i += 1
    f.close()
    np.save("results.npy",results)                                             # save results for testing and evaluation    


    

if __name__ == "__main__":    

    #databasePath = "database_recordings/"
    #fingerprintsPath = "fingerprints/"
    #queryPath = "query_recordings/"
    #queryPath = "database_recordings/"
    #output = "output.txt"
    
    #fingerprintBuilder(databasePath,fingerprintsPath)
    
    #a = time.time()
    audioIdentification(queryPath,fingerprintsPath,"output.txt")
    #b = time.time()
    #print("Average query time =", (b-a)/20)        # Find average query time
    #main.execute()                                 # evaluate 
