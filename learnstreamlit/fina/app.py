import streamlit as st
import pandas as pd
import numpy as np
import time
import company.comp as company
import matplotlib.pyplot as plt
import matplotlib.dates

def main():
  # st.title("A simple financial analysis app")
    st.set_page_config(page_title="Financial Analysis App", layout="wide")
    st.subheader("Let's do some analysis!")

    scr_file = st.file_uploader("Upload your file", type=[ 'xlsx'])
    submit = st.button("Extract Data")




    if submit and scr_file:
        df = pd.read_excel(scr_file,sheet_name='Data Sheet')
        if df.empty:
            st.write("No relavent data found!!!")
            return
        else:
          #st.dataframe(df)
          st.write(df.describe())
          #st.line_chart(df)
          #df.columns[1]
          bus1 = company.Company(df.columns[1])
          st.write(bus1)
          bus1.set_no_of_shares(5700000)
          bus1.set_cmp(2528)
          bus1.process_input_data(df)
          SalesDataFrame  = bus1.get_sales_data()



          st.write("Sales of the company")
          st.bar_chart(SalesDataFrame, x="Report Date", y="Sales")

          st.write("Sales Growth Rate of the company")
          st.bar_chart(SalesDataFrame, x="Report Date", y="Growth_Rate")

          st.write("Raw Material cost of the company")
          st.bar_chart(SalesDataFrame, x="Report Date", y="Raw Material Cost")
          st.write("Change in Inventory of the company")
          st.bar_chart(SalesDataFrame, x="Report Date", y="Change in Inventory")
          st.write("Power and Fuel cost of the company")
          st.bar_chart(SalesDataFrame, x="Report Date", y="Power and Fuel")
          st.write("Other Manufacturing Expenses of the company")
          st.bar_chart(SalesDataFrame, x="Report Date", y="Other Mfr. Exp")

          st.write("Employee Cost of the company")
          st.bar_chart(SalesDataFrame, x="Report Date", y="Employee Cost")

          st.write("Selling and Admin Expenses of the company")
          st.bar_chart(SalesDataFrame, x="Report Date", y="Selling and admin")

          st.write("Other Expenses of the company")
          st.bar_chart(SalesDataFrame, x="Report Date", y="Other Expenses")

          st.write("Operating Profit of the company")
          st.bar_chart(SalesDataFrame, x="Report Date", y="Operating Profit")

          st.write("Operating Profit Margin of the company")
          st.bar_chart(SalesDataFrame, x="Report Date", y="Operating Profit Margin")


          st.line_chart(SalesDataFrame, x="Report Date", y="Sales")

          st.write("quantitative_analysis of the company")
          bus1.quantitative_analysis()
          st.line_chart(SalesDataFrame, x="Report Date", y="EBITDA Margin")


          '''
          ToDO:
          1. Idenfity the growth of Sales rate. How fast its growing. is it sustainable?
          2. Operating margin. Find the companies in the sector, what is their operating margin.
          3. Interest rate. Find the what rate interest rates are being calculated. It should be done
          by using total debt and the rate.
          What is capitalizing the interest cost? analyze and provide output.
          4. Tax rate. What is the tax rate? usually its between 30-34% in india.
            If deviation find out why?
            Tax rate = Tax/Sales
          '''

if __name__ == '__main__':
    main()