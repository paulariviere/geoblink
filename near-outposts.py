#!/usr/bin/env python
import psycopg2 as pg
import sys, os
import numpy as np
import pandas as pd
import pandas.io.sql as psql
from sklearn import linear_model
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression

# Connection to the database. This is for the Postgres server located at localhost
connection = pg.connect("dbname=geoblink user=postgres password=postgres")

# Get ids and positions of the evil outposts from the database
data = psql.read_sql_query("SELECT es_id,es_x,es_y  FROM ES", connection)
ids = data.values[:,0]
positions = data.values[:,1:3]

# Check that we got it right
print("First evil outpost: " + str(ids[0]) + " " + str(positions[0,0]) + " " + str(positions[0,1]))

# We will put the data of the evil outposts within a radius of 15k GUs in table NEO:
cursor = connection.cursor()
sqlInsert = "UPDATE NEO set neo_num = %s where neo_id = %s"

# Initialize array
numOutposts = ids.shape[0]
print("Working with " + str(numOutposts) + " outposts")

# Calculate distance between outposts
print("Calculating distances between outposts")
nearOutposts = np.zeros((numOutposts))

# Loop over outposts
for index, item in enumerate(positions):
    # Initialize the number of near outposts for each outpost
    numNearOutposts = 0
    
    # Run over every other outpost
    for index2, item2 in enumerate(positions):
        if (index != index2):
            # Calculate distance
           distance = np.sqrt(np.sum((positions[index,:]-positions[index2,:])**2, axis=0))
           # Store if it's smaller than 15k GUs
           if (distance < 15000):
               numNearOutposts = numNearOutposts + 1;

    # Assign the number of near outposts at the array
    nearOutposts[index] = numNearOutposts
    # Print on console
    print("There are " + str(numNearOutposts) + " outposts near outpost " + str(ids[index]));
    # Insert in table NP
    cursor.execute(sqlInsert,(numNearOutposts,ids[index]))

# Commit transaction
connection.commit()
