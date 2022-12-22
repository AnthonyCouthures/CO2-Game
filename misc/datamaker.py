import pandas as pd 

AREA = pd.read_csv("data/land-area-km.csv", dtype={"Entity" : "string", "Code" : "string" ,"Year" : int , "Land area (sq. km)" :float} )
POPULATION = pd.read_csv("data/world-population-by-world-regions-post-1820.csv", dtype={"Entity" : "string", "Code" : "string" ,"Year" : int , "Population (historical estimates)" :float})
CO2 = pd.read_csv("data/annual-co-emissions-by-region.csv", dtype={"Entity" : "string", "Code" : "string" ,"Year" : int , "Annual CO2 emissions (zero filled)" :float})

first_year_emission = 2000

CONTRIES_OF_STUDY = ['China', 'Russia', 'India', 'European Union', 'United States', 'Japan'] #,'Africa']

list_data_CO2 = []
list_data_CO2_2020 = []
list_linear_approx = []

list_data_area = []
list_data_population = []

df_data_of_interest = pd.DataFrame(data= { 'Entity' : CONTRIES_OF_STUDY})

for contry in CONTRIES_OF_STUDY:
    # Contry = CO2.loc[CO2['Entity']==contry]
    # index = Contry.loc[Contry['Year'] >= first_year_emission].index
    # array = Contry.loc[index]['Annual CO2 emissions (zero filled)'].values
    # list_data_CO2.append(array)

    # list_linear_approx.append(np.polyfit(Contry['Year'][index], Contry['Annual CO2 emissions (zero filled)'][index],degree_regeression) / 1E9)
    print('Contry :', contry)
    
    Contry = CO2.loc[CO2['Entity']==contry]
    index = Contry.loc[Contry['Year'] == 2020].index
    array = Contry.loc[index]['Annual CO2 emissions (zero filled)'].values / 1E9
    list_data_CO2_2020.append(array[0])



    Contry = AREA.loc[AREA['Entity']==contry]
    index = Contry.loc[Contry['Year'] == 2018].index
    array = Contry.loc[index]['Land area (sq. km)'].values
    list_data_area.append(array[0])

    Contry = POPULATION.loc[POPULATION['Entity']==contry]
    index = Contry.loc[Contry['Year'] == 2020].index
    array = Contry.loc[index]['Population (historical estimates)'].values
    list_data_population.append(array[0])

sum_area = sum(list_data_area)
sum_population = sum(list_data_population)
print(sum_population)
df_data_of_interest = pd.DataFrame(data= { 'Entity' : CONTRIES_OF_STUDY,
                                            'Land area (sq. km)' : list_data_area, 'Land ratio' : list_data_area/ sum_area,
                                            'Population 2020' : list_data_population, 'Population 2020 ratio' : list_data_population/sum_population,
                                            'CO2 emission in Gt 2020' : list_data_CO2_2020 ,
                                            # 'list CO2 emission in tons from {}'.format(first_year_emission) : list_data_CO2
                                            
                                            
                                            })

df_data_of_interest.to_csv('data/data_of_interest.csv', index=False)
df_data_of_interest