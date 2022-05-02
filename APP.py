from urllib.request import urlopen
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import os
import pytz
import numpy as np
import folium
#import sys
from functools import reduce
#import seaborn as sns
from streamlit_folium import folium_static
#to get rid of deprecation warnings
import warnings
warnings.filterwarnings("ignore") 


st.title('Nickel Production from Laterites vs Sulfides')

year = st.slider('year', 1999, 2018)

#for 1999-2003
url1 = "https://s3-us-west-2.amazonaws.com/prd-wret/assets/palladium/production/mineral-pubs/nickel/nickemyb03.xls" 
#for 2004-2008
url2 = "https://s3-us-west-2.amazonaws.com/prd-wret/assets/palladium/production/mineral-pubs/nickel/myb1-2008-nicke.xls"
#for 2009-2013
url3 = "https://s3-us-west-2.amazonaws.com/prd-wret/assets/palladium/production/mineral-pubs/nickel/myb1-2013-nicke.xls"
#for 2014-2018
url4 = "https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/myb1-2018-nicke-adv.xlsx"

#For mine production data
#1999-2003
add1 = pd.read_excel(url1, sheet_name = "Table10", skiprows = 5)
add1.drop(add1.columns[add1.columns.str.contains('Unnamed',case = False)],axis = 1, inplace = True)
#excludes everything in the description part
thre1 = add1['Country']=='Of which:'
h = thre1[thre1].index.values
add1 = add1.iloc[:h[0]-1]

#China: sulfide
check = add1["Country"].str.contains("China") == True
for i in range(len(add1)):
    if check[i] == True:
        add1['Country'][i] = 'China, concentrate'

#creates dataframe for laterite data with zeros only
add1_lat = pd.DataFrame(np.zeros((len(add1.index), len(add1.columns))),columns = ['Country','1999','2000','2001','2002','2003'])
#bool list whether it says laterite or not - if it says ore it is laterite
lat_check = add1["Country"].str.contains("ore") == True
#if country includes "laterite", transfer values in other columns
for i in range(len(add1)):
    if lat_check[i] == True:
        add1_lat['1999'][i] = add1['1999'][i]
        add1_lat['2000'][i] = add1['2000'][i]
        add1_lat['2001'][i] = add1['2001'][i]
        add1_lat['2002'][i] = add1['2002'][i]
        add1_lat['2003'][i] = add1['2003'][i]

#FINDING SULFIDES
add1_sulf = pd.DataFrame(np.zeros((len(add1.index), len(add1.columns))),columns = ['Country','1999','2000','2001','2002','2003'])

#bool list whether it says laterite or not #if country includes "concentrate", transfer values in other columns
sulf_check = add1["Country"].str.contains("concentrate") == True
for i in range(len(add1)):
    if sulf_check[i] == True:
        add1_sulf['1999'][i] = add1['1999'][i]
        add1_sulf['2000'][i] = add1['2000'][i]
        add1_sulf['2001'][i] = add1['2001'][i]
        add1_sulf['2002'][i] = add1['2002'][i]
        add1_sulf['2003'][i] = add1['2003'][i]

#Only takes the country name
for i in range(len(add1)):
    add1["Country"][i] = add1["Country"][i].split(', ', 1)[0]
#transfer country names
add1_sulf['Country'] = add1['Country']
add1_lat['Country'] = add1['Country']

add1.set_index('Country', inplace=True)
add1_sulf.set_index('Country', inplace=True)
add1_lat.set_index('Country', inplace=True)

#Australia: half sulfide, half laterite (assumption)
add1_sulf.loc['Australia'] = add1.loc['Australia'] / 2
add1_lat.loc['Australia'] = add1.loc['Australia'] / 2


#drops rows of all 0s
add1_sulf = add1_sulf.loc[(add1_sulf != 0).any(axis=1)]
add1_lat = add1_lat.loc[(add1_lat != 0).any(axis=1)]
#replaces --s with 0
add1_sulf = add1_sulf.replace('--', 0)
add1_lat = add1_lat.replace('--', 0)

#2004-2008
add2 = pd.read_excel(url2, sheet_name = "Table10", skiprows = 5)
add2.rename(columns = {'Country and products2': 'Country'}, inplace = True)
add2.drop(add2.columns[add2.columns.str.contains('Unnamed',case = False)],axis = 1, inplace = True)

#excludes everything in the description part
thre = add2['Country']=='Of which:'
s = thre[thre].index.values
add2 = add2.iloc[:s[0]-1]

#FINDING LATERITES
#creates dataframe for laterite data with zeros only
add2_lat = pd.DataFrame(np.zeros((len(add2.index), len(add2.columns))),columns = ['Country','2004','2005','2006','2007','2008'])

#Cuba: ammoniacal luqior is laterite
lat_check = add2["Country"].str.contains("ammoniacal liquor5") == True
for i in range(len(add2)):
    if lat_check[i] == True:
        add2['Country'][i] = 'Cuba, ore'
#Macedonia: laterite
lat_check = add2["Country"].str.contains("ferronickel") == True
for i in range(len(add2)):
    if lat_check[i] == True:
        add2['Country'][i] = 'Macedonia, ore'
#philippines
lat_check = add2["Country"].str.contains("Philippines") == True
for i in range(len(add2)):
    if lat_check[i] == True:
        add2['Country'][i+1] = 'Philippines, ore'
        add2["Country"][i+2] = 'Philippines, concentrate'
#Russia
lat_check = add2["Country"].str.contains("Russia") == True
for i in range(len(add2)):
    if lat_check[i] == True:
        add2['Country'][i+1] = 'Russia, ore'
        add2["Country"][i+2] = 'Russia, concentrate'
#China: sulfide
lat_check = add2["Country"].str.contains("China") == True
for i in range(len(add2)):
    if lat_check[i] == True:
        add2['Country'][i] = 'China, concentrate'

#bool list whether it says laterite or not - if it says ore it is laterite
lat_check = add2["Country"].str.contains("ore") == True
#if country includes "laterite", transfer values in other columns
for i in range(len(add2)):
    if lat_check[i] == True:
        add2_lat['2004'][i] = add2['2004'][i]
        add2_lat['2005'][i] = add2['2005'][i]
        add2_lat['2006'][i] = add2['2006'][i]
        add2_lat['2007'][i] = add2['2007'][i]
        add2_lat['2008'][i] = add2['2008'][i]

#FINDING SULFIDES
add2_sulf = pd.DataFrame(np.zeros((len(add2.index), len(add2.columns))),columns = ['Country','2004','2005','2006','2007','2008'])

#bool list
sulf_check = add2["Country"].str.contains("concentrate") == True
for i in range(len(add2)):
    if sulf_check[i] == True:
        add2_sulf['2004'][i] = add2['2004'][i]
        add2_sulf['2005'][i] = add2['2005'][i]
        add2_sulf['2006'][i] = add2['2006'][i]
        add2_sulf['2007'][i] = add2['2007'][i]
        add2_sulf['2008'][i] = add2['2008'][i]

#Only takes the country name
for i in range(len(add2)):
    add2["Country"][i] = add2["Country"][i].split(', ', 1)[0]
#transfer country names
add2_sulf['Country'] = add2['Country']
add2_lat['Country'] = add2['Country']

#drop 'total' rows
add2_sulf.set_index('Country', inplace=True)
add2_sulf.drop(index = 'Total',axis=0,inplace=True)
add2_lat.set_index('Country', inplace=True)
add2_lat.drop(index = 'Total',axis=0,inplace=True)
add2.set_index('Country', inplace=True)

#special conditions for certain countries:
#Australia: half sulfide, half laterite (assumption)
add2_sulf.loc['Australia'] = add2_sulf.loc['Australia'] / 2
add2_lat.loc['Australia'] = add2_lat.loc['Australia'] / 2

#drops rows of all 0s
add2_sulf = add2_sulf.loc[(add2_sulf != 0).any(axis=1)]
add2_lat = add2_lat.loc[(add2_lat != 0).any(axis=1)]

#replaces --s with 0
add2_sulf = add2_sulf.replace('--', 0)
add2_lat = add2_lat.replace('--', 0)

#2009-2013
add3 = pd.read_excel(url3, sheet_name = "T10", skiprows = 5)
add3.rename(columns = {'Country and products3': 'Country'}, inplace = True)
add3.drop(add3.columns[add3.columns.str.contains('Unnamed',case = False)],axis = 1, inplace = True)
#excludes everything in the description part
thre = add3['Country']=='Of which:'
s = thre[thre].index.values
add3 = add3.iloc[:s[0]-1]

#FINDING LATERITES
#creates dataframe for laterite data with zeros only
add3_lat = pd.DataFrame(np.zeros((len(add3.index), len(add3.columns))),columns = ['Country','2009','2010','2011','2012','2013'])

#Cuba: laterite
lat_check = add3["Country"].str.contains("Cuba") == True
for i in range(len(add3)):
    if lat_check[i] == True:
        add3['Country'][i+3] = 'Cuba, ore'
#Macedonia: laterite
lat_check = add3["Country"].str.contains("ferronickel") == True
for i in range(len(add3)):
    if lat_check[i] == True:
        add3['Country'][i] = 'Macedonia, ore'
#philippines
lat_check = add3["Country"].str.contains("Philippines") == True
for i in range(len(add3)):
    if lat_check[i] == True:
        add3['Country'][i+1] = 'Philippines, ore'
        add3["Country"][i+2] = 'Philippines, concentrate'
#Russia
lat_check = add3["Country"].str.contains("Russia") == True
for i in range(len(add3)):
    if lat_check[i] == True:
        add3['Country'][i+1] = 'Russia, ore'
        add3["Country"][i+2] = 'Russia, concentrate'
#China: sulfide
lat_check = add3["Country"].str.contains("China") == True
for i in range(len(add3)):
    if lat_check[i] == True:
        add3['Country'][i] = 'China, concentrate'

#bool list whether it says laterite or not - if it says ore it is laterite
lat_check = add3["Country"].str.contains("ore") == True
#if country includes "laterite", transfer values in other columns
for i in range(len(add3)):
    if lat_check[i] == True:
        add3_lat['2009'][i] = add3['2009'][i]
        add3_lat['2010'][i] = add3['2010'][i]
        add3_lat['2011'][i] = add3['2011'][i]
        add3_lat['2012'][i] = add3['2012'][i]
        add3_lat['2013'][i] = add3['2013'][i]

#FINDING SULFIDES
add3_sulf = pd.DataFrame(np.zeros((len(add3.index), len(add3.columns))),columns = ['Country','2009','2010','2011','2012','2013'])

#bool list 
sulf_check = add3["Country"].str.contains("concentrate") == True
for i in range(len(add3)):
    if sulf_check[i] == True:
        add3_sulf['2009'][i] = add3['2009'][i]
        add3_sulf['2010'][i] = add3['2010'][i]
        add3_sulf['2011'][i] = add3['2011'][i]
        add3_sulf['2012'][i] = add3['2012'][i]
        add3_sulf['2013'][i] = add3['2013'][i]

#Only country name
for i in range(len(add3)):
    add3["Country"][i] = add3["Country"][i].split(', ', 1)[0]
#transfer country names
add3_sulf['Country'] = add3['Country']
add3_lat['Country'] = add3['Country']

#drop 'total' rows
add3_sulf.set_index('Country', inplace=True)
add3_sulf.drop(index = 'Total',axis=0,inplace=True)
add3_lat.set_index('Country', inplace=True)
add3_lat.drop(index = 'Total',axis=0,inplace=True)
add3.set_index('Country', inplace=True)

#special conditions for certain countries:
#Australia: half sulfide, half laterite (assumption)
add3_sulf.loc['Australia'] = add3_sulf.loc['Australia'] / 2
add3_lat.loc['Australia'] = add3_lat.loc['Australia'] / 2

#drops rows of all 0s
add3_sulf = add3_sulf.loc[(add3_sulf != 0).any(axis=1)]
add3_lat = add3_lat.loc[(add3_lat != 0).any(axis=1)]

#replaces --s with 0
add3_sulf = add3_sulf.replace('--', 0)
add3_lat = add3_lat.replace('--', 0)

#2014-2018
add4 = pd.read_excel(url4, sheet_name = "T10", skiprows = 5)
add4.rename(columns = {'Country or locality3': 'Country'}, inplace = True)
add4.columns = add4.columns.astype(str)
add4.drop(add4.columns[add4.columns.str.contains('Unnamed',case = False)],axis = 1, inplace = True)

#excludes everything in the description part
thre = add4['Country']=='Of which:'
s = thre[thre].index.values
add4 = add4.iloc[:s[0]-1]
#FINDING LATERITES
#creates dataframe for laterite data with zeros only
add4_lat = pd.DataFrame(np.zeros((len(add4.index), len(add4.columns))),columns = ['Country','2014','2015','2016','2017','2018'])

#Cuba: laterite
lat_check = add4["Country"].str.contains("Brazil") == True
for i in range(len(add4)):
    if lat_check[i] == True:
        add4['Country'][i] = 'Brazil, laterite'
#Colombia: laterite
lat_check = add4["Country"].str.contains("Colombia") == True
for i in range(len(add4)):
    if lat_check[i] == True:
        add4['Country'][i+2] = 'Colombia, laterite'
#Finland: sulfide
lat_check = add4["Country"].str.contains("Finland") == True
for i in range(len(add4)):
    if lat_check[i] == True:
        add4['Country'][i] = 'Finland, sulfide'
#Russia
lat_check = add4["Country"].str.contains("Russia") == True
for i in range(len(add4)):
    if lat_check[i] == True:
        add4['Country'][i+1] = 'Russia, laterite'
        add4["Country"][i+2] = 'Russia, sulfide'
#China: sulfide
lat_check = add4["Country"].str.contains("China") == True
for i in range(len(add4)):
    if lat_check[i] == True:
        add4['Country'][i] = 'China, sulfide'

#bool list whether it says laterite or not - if it says ore it is laterite
lat_check = add4["Country"].str.contains("laterite") == True
#if country includes "laterite", transfer values in other columns
for i in range(len(add4)):
    if lat_check[i] == True:
        add4_lat['2014'][i] = add4['2014'][i]
        add4_lat['2015'][i] = add4['2015'][i]
        add4_lat['2016'][i] = add4['2016'][i]
        add4_lat['2017'][i] = add4['2017'][i]
        add4_lat['2018'][i] = add4['2018'][i]

#FINDING SULFIDES
add4_sulf = pd.DataFrame(np.zeros((len(add4.index), len(add4.columns))),columns = ['Country','2014','2015','2016','2017','2018'])
#bool list
sulf_check = add4["Country"].str.contains("sulfide") == True
for i in range(len(add4)):
    if sulf_check[i] == True:
        add4_sulf['2014'][i] = add4['2014'][i]
        add4_sulf['2015'][i] = add4['2015'][i]
        add4_sulf['2016'][i] = add4['2016'][i]
        add4_sulf['2017'][i] = add4['2017'][i]
        add4_sulf['2018'][i] = add4['2018'][i]

#Only country name
for i in range(len(add4)):
    add4["Country"][i] = add4["Country"][i].split(', ', 1)[0]
#transfer country names
add4_sulf['Country'] = add4['Country']
add4_lat['Country'] = add4['Country']

#sets index
add4_sulf.set_index('Country', inplace=True)
add4_lat.set_index('Country', inplace=True)
add4.set_index('Country', inplace=True)

#Australia: half sulfide, half laterite (assumption)
add4_sulf.loc['Australia'] = add4.loc['Australia'] / 2
add4_lat.loc['Australia'] = add4.loc['Australia'] / 2

add4_lat = add4_lat.replace(np.nan, 0)
add4_sulf = add4_sulf.replace(np.nan, 0)
add4_lat = add4_lat.replace('NA ', 0)
add4_sulf = add4_sulf.replace('NA ', 0)

#drops rows of all 0s
add4_sulf = add4_sulf.loc[(add4_sulf != 0).any(axis=1)]
add4_lat = add4_lat.loc[(add4_lat != 0).any(axis=1)]

#replaces --s with 0
add4_sulf = add4_sulf.replace('--', 0)
add4_lat = add4_lat.replace('--', 0)

#MERGING
data_frames_lat = [add1_lat, add2_lat, add3_lat, add4_lat]
merged_lat = reduce(lambda  left,right: pd.merge(left,right,on=['Country'],how='outer'), data_frames_lat)
merged_lat = merged_lat.replace(np.nan, 0)

data_frames_sulf = [add1_sulf, add2_sulf, add3_sulf, add4_sulf]
merged_sulf = reduce(lambda  left,right: pd.merge(left,right,on=['Country'],how='outer'), data_frames_sulf)
merged_sulf = merged_sulf.replace(np.nan, 0)

#plot the sums of all laterite and sulfides with changing year
merged_lat["sum"] = merged_lat.sum(axis=1)
merged_sulf["sum"] = merged_sulf.sum(axis=1)

#convert index column into a year column
merged_lat.reset_index(inplace=True)
merged_lat = merged_lat.rename(columns = {'index':'Year'})
merged_lat = merged_lat.sort_values(by=['sum'], ascending=False)
lat_plot = merged_lat.iloc[0:8]

merged_sulf.reset_index(inplace=True)
merged_sulf = merged_sulf.rename(columns = {'index':'Year'})
merged_sulf = merged_sulf.sort_values(by=['sum'], ascending=False)
sulf_plot = merged_sulf.iloc[0:8]

#Taking coordinates of countries from web and merging them with our data
#https://www.kaggle.com/datasets/paultimothymooney/latitude-and-longitude-for-every-country-and-state/download
df = pd.read_csv('world_country_and_usa_states_latitude_and_longitude_values.csv')

cnt = df[['country','latitude','longitude']]
cnt = cnt.rename(columns={"country": "Country"})
merged_lat=pd.merge(merged_lat,cnt, on='Country', how='inner')
merged_sulf=pd.merge(merged_sulf,cnt, on='Country', how='inner')

y = str(year)
# combined map
c = folium.Map(tiles="Stamen Terrain")
for index, row in merged_lat.iterrows():
    folium.Circle(location=[row['latitude'], row['longitude']],
          popup= 'Nickel production:' +str(row[y]),
          tooltip=row['Country'],
          radius=row[y]*4,
          color='darkred',
          fill=True,fill_color='orange').add_to(c)
for index, row in merged_sulf.iterrows():
    folium.Circle(location=[row['latitude'], row['longitude']],
          popup= 'Nickel production:' +str(row[y]),
          tooltip=row['Country'],
          radius=row[y]*4,
          color='darkblue',
          fill=True,fill_color='mediumblue').add_to(c)

folium_static(c)