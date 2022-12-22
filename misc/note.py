USA	 =  ['United States']
CAN	 =  ['Canada']
JPK  =  ['Japan', 'South Korea']
ANZ  =  ['Australia', 'New Zealand']
CEE  =  ['Albania', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Czech Republic', 'Hungary', 'FYR Macedonia', 'Poland', 'Romania', 'Slovakia', 'Slovenia', 'Yugoslavia']
FSU	 =	['Armenia', 'Azerbaijan', 'Belarus', 'Estonia', 'Georgia', 'Kazakhstan', 'Kyrgyzstan', 'Latvia', 'Lithuania', 'Moldova', 'Russia', 'Tajikistan', 'Turkmenistan', 'Ukraine', 'Uzbekistan']
MDE	 = 	['Bahrain', 'Iran', 'Iraq', 'Israel', 'Jordan', 'Kuwait', 'Lebanon', 'Oman', 'Qatar', 'Saudi Arabia', 'Syria', 'Turkey', 'United Arab Emirates', 'West Bank and Gaza', 'Yemen']
CAM	 = 	['Belize', 'Costa Rica', 'El Salvador', 'Guatemala', 'Honduras', 'Mexico', 'Nicaragua', 'Panama']
SAM	 =	['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'French Guiana', 'Guyana', 'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela']
SAS	 =  ['Afghanistan', 'Bangladesh', 'Bhutan', 'India', 'Nepal', 'Pakistan', 'Sri Lanka']
SEA	 =	['Brunei', 'Cambodia', 'East Timor', 'Indonesia', 'Laos', 'Malaysia', 'Myanmar', 'Papua New Guinea', 'Philippines', 'Singapore', 'Taiwan', 'Thailand', 'Vietnam']
CHI	 =  ['China', 'Hong Kong', 'North Korea', 'Macau', 'Mongolia']
NAF	 =	['Algeria', 'Egypt', 'Libya', 'Morocco', 'Tunisia', 'Western Sahara']
SSA	 =	['Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon', 'Cape Verde', 'Central African Republic', 'Chad', 'Congo-Brazzaville', 'Congo-Kinshasa', 'Cote d’Ivoire', 'Djibouti', 'Equatorial Guinea', 'Eritrea', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea- Bissau', 'Kenya', 'Lesotho', 'Liberia', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'Senegal', 'Sierra Leone', 'Somalia', 'South Africa', 'Sudan', 'Swaziland', 'Tanzania', 'Togo', 'Uganda', 'Zambia', 'Zimbabwe']
SIS	 =  ['Antigua and Barbuda', 'Aruba', 'Bahamas', 'Barbados', 'Bermuda', 'Comoros', 'Cuba', 'Dominica', 'Dominican Republic', 'Fiji', 'French Polynesia', 'Grenada', 'Guadeloupe', 'Haiti', 'Jamaica', 'Kiribati', 'Maldives', 'Marshall Islands', 'Martinique', 'Mauritius', 'Micronesia', 'Nauru', 'Netherlands Antilles', 'New Caledonia', 'Palau', 'Puerto Rico', 'Reunion', 'Samoa', 'Sao Tome and Principe', 'Seychelles', 'Solomon Islands', 'St Kitts and Nevis', 'St Lucia', 'St Vincent and Grenadines', 'Tonga', 'Trinidad and Tobago', 'Tuvalu', 'Vanuatu', 'Virgin Islands']
WEU  =  ['Andorra', 'Austria', 'Belgium', 'Cyprus', 'Denmark', 'Finland', 'France', 'Germany', 'Greece', 'Iceland', 'Ireland', 'Italy', 'Liechtenstein', 'Luxembourg', 'Malta', 'Monaco', 'Netherlands', 'Norway', 'Portugal', 'San Marino', 'Spain', 'Sweden', 'Switzerland', 'United Kingdom']

import inspect

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


import pandas as pd
list_label = [USA, CAN, JPK, ANZ, CEE, FSU, MDE, CAM, SAM, SAS, SEA, CHI, NAF, SSA, SIS, WEU]

for label in list_label:
    data_impact = pd.read_csv('data\impact factor.csv')
    impact_factor = data_impact[data_impact[0] == retrieve_name(label)[0]]
    print(impact_factor)