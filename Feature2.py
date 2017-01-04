import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans

from math import sqrt

style.use('ggplot')

df_train=pd.read_csv('train.txt', header=0, delimiter="\t", quoting=3)
#Sort dataframe by time
df = df_train.sort_values(["time"], ascending=[1])

def handle_non_numerical_data(df):
	columns = df.columns.values
	
	for column in columns:
		text_digit_vals = {}
		def convert_to_int(val):
			return text_digit_vals[val]
			
		if df[column].dtype != np.int64 and df[column].dtype != np.float64:
			column_contents = df[column].values.tolist()
			unique_elements = set(column_contents)
			x = 0
			for unique in unique_elements:
				if unique not in text_digit_vals:
					text_digit_vals[unique] = x
					x+=1
					
			df[column] = list(map(convert_to_int, df[column]))
			
	return df
	
#Convert id into a number
df = handle_non_numerical_data(df)

#Converting unix timestamp to readable date
df['time'] = pd.to_datetime(df['time'],unit='s')

df_array_temp = {}
df_array = {}

number_of_elem = df.id.unique()

for i in range(len(number_of_elem)):
    #Split in dataframes grouped by same id
    df_array_temp[i] = df[df['id'] == i]
    df_array_temp[i] = df_array_temp[i].reset_index(drop=True)
    df_array_temp[i].drop(['id'], 1, inplace=True)
    #Split in dataframes grouped by same day
    df_array[i] = {}
    df_array[i][0] = df_array_temp[i][df_array_temp[i]['time'].dt.date == datetime.date(year=2015,month=11,day=9)]
    df_array[i][0] = df_array[i][0].reset_index(drop=True)
    df_array[i][1] = df_array_temp[i][df_array_temp[i]['time'].dt.date == datetime.date(year=2015,month=11,day=10)]
    df_array[i][1] = df_array[i][1].reset_index(drop=True)
    df_array[i][2] = df_array_temp[i][df_array_temp[i]['time'].dt.date == datetime.date(year=2015,month=11,day=11)]
    df_array[i][2] = df_array[i][2].reset_index(drop=True)
    df_array[i][3] = df_array_temp[i][df_array_temp[i]['time'].dt.date == datetime.date(year=2015,month=11,day=12)]
    df_array[i][3] = df_array[i][3].reset_index(drop=True)
    df_array[i][4] = df_array_temp[i][df_array_temp[i]['time'].dt.date == datetime.date(year=2015,month=11,day=13)]
    df_array[i][4] = df_array[i][4].reset_index(drop=True)
    df_array[i][5] = df_array_temp[i][df_array_temp[i]['time'].dt.date == datetime.date(year=2015,month=11,day=14)]
    df_array[i][5] = df_array[i][5].reset_index(drop=True)
    df_array[i][6] = df_array_temp[i][df_array_temp[i]['time'].dt.date == datetime.date(year=2015,month=11,day=15)]
    df_array[i][6] = df_array[i][6].reset_index(drop=True)


stop_x = [11038.08464497, 3425.67079005, 283.08479678]
stop_y = [8253.17542416, 3469.94198377, 163.45489494]

coordinates = pd.DataFrame()

#51 element, for each found the route by looking at the map
#For each day we will take only one route
#Loosing part of data, but I guess the remaining data will be enough to guess stop stations
for j in range(len(number_of_elem)):
    for q in range(7):
        if df_array[j][q].empty:
            #Might be empty
            continue
        k = -1
        l = -1
        start = 999999
        end = 999999
        for i in range(160):
            if(i > len(df_array[j][q]) - 1):
                break
            if (sqrt((df_array[j][q]["x"][i]-stop_x[0])*(df_array[j][q]["x"][i]-stop_x[0]) + (df_array[j][q]["y"][i]-stop_y[0])*(df_array[j][q]["y"][i]-stop_y[0])) < start):
                start = sqrt((df_array[j][q]["x"][i]-stop_x[0])*(df_array[j][q]["x"][i]-stop_x[0]) + (df_array[j][q]["y"][i]-stop_y[0])*(df_array[j][q]["y"][i]-stop_y[0]))
                k = i
            if (sqrt((df_array[j][q]["x"][i]-stop_x[2])*(df_array[j][q]["x"][i]-stop_x[2]) + (df_array[j][q]["y"][i]-stop_y[2])*(df_array[j][q]["y"][i]-stop_y[2])) < end):
                end = sqrt((df_array[j][q]["x"][i]-stop_x[2])*(df_array[j][q]["x"][i]-stop_x[2]) + (df_array[j][q]["y"][i]-stop_y[2])*(df_array[j][q]["y"][i]-stop_y[2]))
                l = i
        if (k > l):
            end = 999999
            for i in (list(set(range(160)) - set(range(k)))):
                if(i > len(df_array[j][q]) - 1):
                    break
                if (sqrt((df_array[j][q]["x"][i]-stop_x[2])*(df_array[j][q]["x"][i]-stop_x[2]) + (df_array[j][q]["y"][i]-stop_y[2])*(df_array[j][q]["y"][i]-stop_y[2])) < end):
                    end = sqrt((df_array[j][q]["x"][i]-stop_x[2])*(df_array[j][q]["x"][i]-stop_x[2]) + (df_array[j][q]["y"][i]-stop_y[2])*(df_array[j][q]["y"][i]-stop_y[2]))
                    l = i
        if (l == 159):
            end = 999999
            for i in (list(set(range(210)) - set(range(160)))):
                if(i > len(df_array[j][q]) - 1):
                    break
                if (sqrt((df_array[j][q]["x"][i]-stop_x[2])*(df_array[j][q]["x"][i]-stop_x[2]) + (df_array[j][q]["y"][i]-stop_y[2])*(df_array[j][q]["y"][i]-stop_y[2])) < end):
                    end = sqrt((df_array[j][q]["x"][i]-stop_x[2])*(df_array[j][q]["x"][i]-stop_x[2]) + (df_array[j][q]["y"][i]-stop_y[2])*(df_array[j][q]["y"][i]-stop_y[2]))
                    l = i
        if (l == 209):
            end = 999999
            for i in (list(set(range(240)) - set(range(210)))):
                if(i > len(df_array[j][q]) - 1):
                    break
                if (sqrt((df_array[j][q]["x"][i]-stop_x[2])*(df_array[j][q]["x"][i]-stop_x[2]) + (df_array[j][q]["y"][i]-stop_y[2])*(df_array[j][q]["y"][i]-stop_y[2])) < end):
                    end = sqrt((df_array[j][q]["x"][i]-stop_x[2])*(df_array[j][q]["x"][i]-stop_x[2]) + (df_array[j][q]["y"][i]-stop_y[2])*(df_array[j][q]["y"][i]-stop_y[2]))
                    l = i
        if (l == 239):
            end = 999999
            for i in (list(set(range(300)) - set(range(240)))):
                if(i > len(df_array[j][q]) - 1):
                    break
                if (sqrt((df_array[j][q]["x"][i]-stop_x[2])*(df_array[j][q]["x"][i]-stop_x[2]) + (df_array[j][q]["y"][i]-stop_y[2])*(df_array[j][q]["y"][i]-stop_y[2])) < end):
                    end = sqrt((df_array[j][q]["x"][i]-stop_x[2])*(df_array[j][q]["x"][i]-stop_x[2]) + (df_array[j][q]["y"][i]-stop_y[2])*(df_array[j][q]["y"][i]-stop_y[2]))
                    l = i
        if(k < l):
            for i in range(l - k):
                if((df_array[j][q]["y"][i+k] < stop_y[0]) and (df_array[j][q]["x"][i+k] > stop_x[2])):
                    coordinates = coordinates.append({'x': df_array[j][q]["x"][i+k], 'y': df_array[j][q]["y"][i+k]}, ignore_index=True)


#Adding points to known station so KMeans will guess it as a stop station 
    
for i in range(50):
    coordinates = coordinates.append({'x': stop_x[1], 'y': stop_y[1]}, ignore_index=True)

for i in range(len(coordinates)):
    plt.scatter(coordinates['x'][i], coordinates['y'][i], s=12)
    
plt.scatter(stop_x, stop_y, s=50, c='r')

X = coordinates[['x','y']].as_matrix()

#Using KMeans algorithm
clf = KMeans(n_clusters=37)

clf.fit(X)

#We will take centroids centers as our stop stations
centroids = clf.cluster_centers_

output = pd.DataFrame()

dist = 999999
k = -1
for i in range(len(centroids)):
    if (sqrt((centroids[i][0]-stop_x[0])*(centroids[i][0]-stop_x[0]) + (centroids[i][1]-stop_y[0])*(centroids[i][1]-stop_y[0])) < dist):
        dist = sqrt((centroids[i][0]-stop_x[0])*(centroids[i][0]-stop_x[0]) + (centroids[i][1]-stop_y[0])*(centroids[i][1]-stop_y[0]))
        k = i

output = output.append({'x': centroids[k][0], 'y': centroids[k][1]}, ignore_index=True)

centroids = np.delete(centroids, (k), axis=0)

for j in range(36):
    dist = 999999
    k = -1
    for i in range(len(centroids)):
        if (sqrt((centroids[i][0]-output['x'][j])*(centroids[i][0]-output['x'][j]) + (centroids[i][1]-output['y'][j])*(centroids[i][1]-output['y'][j])) < dist):
            dist = sqrt((centroids[i][0]-output['x'][j])*(centroids[i][0]-output['x'][j]) + (centroids[i][1]-output['y'][j])*(centroids[i][1]-output['y'][j]))
            k = i
    output = output.append({'x': centroids[k][0], 'y': centroids[k][1]}, ignore_index=True)
    centroids = np.delete(centroids, (k), axis=0)

for i in range(len(output)):
    plt.scatter(output['x'][i], output['y'][i], marker='x', s=50, linewidths=2)
            
output.to_csv("output1.txt", index=False)
    
plt.show()
