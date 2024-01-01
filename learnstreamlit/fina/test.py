import pandas as pd

excel_file = 'data.xlsx'

df = pd.read_excel(excel_file,sheet_name='Data Sheet')
#print(df)
print(df.columns)
#print(df.dtypes)
#print(df.describe())
print(df.at[0,'COMPANY NAME'])

#for col in df:
print("here it output")
#print(df[0:1][20:21])

index = 0
col_names = []
dates = []
sales =[]
for col in df.columns[0:11]:
    index = index + 1
    if index == 1:
        report_date = df[col][14]
        sales_data = df[col][15]
        col_names.append(df[col][14]) #Report Data
        col_names.append(df[col][15]) #Sales
        print("What is it:",df[col][38])
        #print("col_name:", col_names)


    else:
        dates.append(df[col][14])
        sales.append(df[col][15])
        #print("sales:", sales)

    #print("index:", index, df[col][14])

#SalesDataFrame = pd.DataFrame(dates,sales,columns=col_names)
SalesDataFrame = pd.DataFrame()
#print(dates[0].year)
dates = [x.year for x in dates]


SalesDataFrame[report_date] = dates
SalesDataFrame[sales_data] = sales
#print(SalesDataFrame)
print("="*50)
#index = 0
#for col in df.columns[0:11]:
#    index = index + 1
#    if index == 1:
#        col_names.append(df[col][15])
#        print("col_name:", col_names)
#
#    else:
#        #sales.append(df[col][15])
#        pass
#