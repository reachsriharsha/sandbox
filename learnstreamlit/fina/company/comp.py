
import pandas as pd
import streamlit as st
class Company():
    def __init__(self, name):
        self.name = name

    def __str__(self) -> str:
        return f"{self.name}"
    
    def set_sales_data(self, sales_data):
        self.sales_data = sales_data

    def get_sales_data(self):
        return self.sales_data
    
    def set_q_sales_data(self, q_sales_data):
        self.q_sales_data = q_sales_data
    
    def get_q_sales_data(self):
        return self.q_sales_data
    
    def set_balance_sheet_data(self, balance_sheet_data):
        self.balance_sheet_data = balance_sheet_data
    
    def get_balance_sheet_data(self):
        return self.balance_sheet_data
    
    def set_cash_flow_data(self, cash_flow_data):
        self.cash_flow_data = cash_flow_data

    def get_cash_flow_data(self):   
        return self.cash_flow_data
    
    def process_input_data(self, df):
        index = 0
        #col_names = []
        dates = []
        sales =[]
        raw_material_cost = []
        change_in_inventory = []
        power_and_fuel = []
        other_mfg_expenses = []
        employee_cost = []
        selling_and_admin = []
        other_exepenses = []
        operating_profit = []
        opm = []
        other_income_data = []
        depreciation_data = []
        interest_data = []
        pbt_data =[]
        tax_data = []
        profit_data =[] #PAT
        net_profit_margin = []
        dividend_data = []

        #Quarterly Data
        qdates = []
        qsales_data = []
        qexpense_data = []
        qother_income_data = []
        qdepreciation_data = []
        qinterest_data = [] 
        qpbt_data = []
        qtax_data = []
        qnet_profit_data = []
        qope_profit_data = []

        #Balance Sheet data
        bal_report_date = []
        bal_equity_share_capital = []
        bal_reserves = []
        bal_borrowings = []
        bal_other_liabilities = []
        bal_tot_liabilities = []
        bal_net_block = []
        bal_cap_wip = []
        bal_investments = []
        bal_other_assets = []
        bal_tot_assets = []
        bal_receivable = []
        bal_inventory = []
        bal_cash_in_bank = []
        bal_no_of_equity_shares = []
        bal_new_bonus_shares = []
        bal_face_value = []     

        #Cash Flow Data
        cf_report_date = []
        cf_cash_from_operating_activity = []    
        cf_cash_from_investing_activity = []
        cf_from_financing_activity = []
        cf_net_cash_flow = []


        for col in df.columns[0:11]:
              index = index + 1
              if index == 1:
                  ##Sales Data name from P&L
                  report_date = df[col][14]
                  sales_data = df[col][15]
                  raw_material = df[col][16]  
                  chg_in_inventory = df[col][17]
                  pwr_and_fuel = df[col][18]
                  otr_mfg_expenses = df[col][19]
                  emp_cost = df[col][20]
                  sell_and_admin = df[col][21]
                  oth_expense = df[col][22]
                  other_income =df[col][23]
                  depreciation = df[col][24]
                  interest_name = df[col][25]
                  pbt_name = df[col][26]
                  tax_name = df[col][27]
                  profit_name = df[col][28] #PAT
                  divident_name = df[col][29] #Divident

                  ##Quarterly Data name
                  q_report_date_name = df[col][39]
                  q_sales_data_name = df[col][40]
                  q_expenses_name = df[col][41]
                  q_other_income_name = df[col][42]
                  q_depreciation_name = df[col][43]
                  q_interest_name = df[col][44]
                  q_pbt_name =df[col][45]
                  q_tax_name = df[col][46]
                  q_net_profit_name = df[col][47]
                  q_op_profit_name = df[col][48]

                  ##Balance Sheet Data
                  bal_report_date_name = df[col][54]
                  bal_equity_share_capital_name = df[col][55]
                  bal_reserves_name = df[col][56]
                  bal_borrowings_name = df[col][57]
                  bal_other_liabilities_name = df[col][58]
                  bal_tot_liabilities_name = df[col][59]

                  bal_net_block_name = df[col][60]
                  bal_cap_wip_name = df[col][61]
                  bal_investments_name = df[col][62]
                  bal_other_assets_name = df[col][63]
                  bal_tot_assets_name = df[col][64]

                  bal_receivable_name = df[col][65]
                  bal_inventory_name = df[col][66]
                  bal_cash_in_bank_name = df[col][67]
                  bal_no_of_equity_shares_name = df[col][68]
                  bal_new_bonus_shares_name = df[col][69]
                  bal_face_value_name = df[col][70]


                  ##Cash Flow Data
                  cf_report_date_name = df[col][79]
                  cf_cash_from_operating_activity_name = df[col][80]
                  cf_cash_from_investing_activity_name = df[col][81]
                  cf_cash_from_financial_activity_name  = df[col][82]
                  cf_net_cash_flow_name = df[col][83]


                  ##Sales Data from P&L
                  #col_names.append(df[col][14]) #Report Date
                  #col_names.append(df[col][15]) #Sales
                  #col_names.append(df[col][16]) #Raw Material Cost
                  #col_names.append(df[col][17]) #Change in Inventory
                  #col_names.append(df[col][18]) #Power and Fuel
                  #col_names.append(df[col][19]) #Other Mfg Expenses
                  #col_names.append(df[col][20]) #Employee Cost
                  #col_names.append(df[col][21]) #Selling and Admin
                  #col_names.append(df[col][22]) #Other Expenses
                  #col_names.append(df[col][23]) #Other Income
                  #col_names.append(df[col][24]) #Depreciation
                  #col_names.append(df[col][25]) #Interest
                  #col_names.append(df[col][26]) #PBT
                  #col_names.append(df[col][27]) #Tax
                  #col_names.append(df[col][28]) #Net profit
                  #col_names.append(df[col][29]) #Divident

                  ##Quarterly Data 



              else:
                   ##Sales Data from P&L
                  dates.append(df[col][14])  #Report Date#Sales
                  sales.append(df[col][15])  #Sales
                  raw_material_cost.append(df[col][16]) #Raw Material Cost
                  change_in_inventory.append(df[col][17]) #Change in Inventory
                  power_and_fuel.append(df[col][18])#Power and Fuel
                  other_mfg_expenses.append(df[col][19]) #Other Mfg Expenses
                  employee_cost.append(df[col][20]) #Employee Cost
                  selling_and_admin.append(df[col][21]) #Selling and Admin
                  other_exepenses.append(df[col][22]) #Other Expenses
                  other_income_data.append(df[col][23]) #Other Income
                  depreciation_data.append(df[col][24]) #Depreciation
                  interest_data.append(df[col][25]) #Interest
                  pbt_data.append(df[col][26]) #PBT
                  tax_data.append(df[col][27]) #Tax data
                  profit_data.append(df[col][28]) #Net Profit 
                  dividend_data.append(df[col][29]) #Divident

                  ##Quarterly Data
                  qdates.append(df[col][39]) #Report Date
                  qsales_data.append(df[col][40]) #Sales
                  qexpense_data.append(df[col][41]) #Expenses
                  qother_income_data.append(df[col][42]) #Other Income
                  qdepreciation_data.append(df[col][43]) #Depreciation
                  qinterest_data.append(df[col][44]) #Interest
                  qpbt_data.append(df[col][45]) #PBT
                  qtax_data.append(df[col][46]) #Tax
                  qnet_profit_data.append(df[col][47]) #Net Profit
                  qope_profit_data.append(df[col][48]) #Operating Profit

                  ##Balance Sheet Data
                  bal_report_date.append(df[col][54])  #Report Date
                  bal_equity_share_capital.append(df[col][55]) #Equity Share Capital
                  bal_reserves.append(df[col][56]) #Reserves
                  bal_borrowings.append(df[col][57]) #Borrowings
                  bal_other_liabilities.append(df[col][58]) #Other Liabilities
                  bal_tot_liabilities.append(df[col][59]) #Total Liabilities
                  bal_net_block.append(df[col][60]) #Net Block
                  bal_cap_wip.append(df[col][61]) #Capital WIP
                  bal_investments.append(df[col][62]) #Investments
                  bal_other_assets.append(df[col][63]) #Other Assets
                  bal_tot_assets.append(df[col][64]) #Total Assets  
                  bal_receivable.append(df[col][65]) #Receivable
                  bal_inventory.append(df[col][66]) #Inventory
                  bal_cash_in_bank.append(df[col][67]) #Cash in Bank
                  bal_no_of_equity_shares.append(df[col][68]) #No of Equity Shares
                  bal_new_bonus_shares.append(df[col][69]) #New Bonus Shares
                  bal_face_value.append(df[col][70]) #Face Value
                  
                  ##Cash Flow Data
                  cf_report_date.append(df[col][79]) #Report Date
                  cf_cash_from_operating_activity.append(df[col][80]) #Cash from Operating Activity
                  cf_cash_from_investing_activity.append(df[col][81]) #Cash from Investing Activity
                  cf_from_financing_activity.append(df[col][82]) #Cash from Financing Activity    
                  cf_net_cash_flow.append(df[col][83]) #Net Cash Flow

        dates = [x.year for x in dates]
        q_yr_month = []
        for i in range(len(dates)):
            #for each year calcuate the total expense to derive the operating profit
            # and Operating profit margin
            dates[i] = str(dates[i])
            operating_profit.append(sales[i] - raw_material_cost[i] + change_in_inventory[i] - ( power_and_fuel[i] + other_mfg_expenses[i] + employee_cost[i] + selling_and_admin[i] + other_exepenses[i]))
            opm.append((operating_profit[i]/sales[i])*100)

            #Quarterly date data
            q_yr_month.append(str(qdates[i].year) + "-" + str(qdates[i].month))

            #Net profit margin NPM
            net_profit_margin.append((profit_data[i]/sales[i])*100)

        SalesDataFrame = pd.DataFrame()

        SalesDataFrame[report_date] = dates
        SalesDataFrame[sales_data] = sales
        SalesDataFrame['Growth_Rate'] = SalesDataFrame[sales_data].pct_change(periods=1) * 100
        SalesDataFrame[raw_material] = raw_material_cost
        SalesDataFrame[chg_in_inventory] = change_in_inventory
        SalesDataFrame[pwr_and_fuel] = power_and_fuel
        SalesDataFrame[otr_mfg_expenses] = other_mfg_expenses 
        SalesDataFrame[emp_cost] = employee_cost
        SalesDataFrame[sell_and_admin] = selling_and_admin
        SalesDataFrame[oth_expense] = other_exepenses
        SalesDataFrame[other_income] = other_income_data
        SalesDataFrame[depreciation] = depreciation_data
        SalesDataFrame['Operating Profit'] = operating_profit
        SalesDataFrame['Operating Profit Margin'] = opm
        SalesDataFrame[interest_name] = interest_data
        SalesDataFrame[pbt_name] = pbt_data
        SalesDataFrame[profit_name] = profit_data #PAT
        SalesDataFrame['Net Profit Margin'] = net_profit_margin #NPM
        SalesDataFrame[divident_name] = dividend_data

        self.set_sales_data(SalesDataFrame)

        QtrDataFrame = pd.DataFrame()
        QtrDataFrame[q_report_date_name] = q_yr_month
        QtrDataFrame[q_sales_data_name] = qsales_data
        QtrDataFrame[q_expenses_name] = qexpense_data   
        QtrDataFrame[q_other_income_name] = qother_income_data
        QtrDataFrame[q_depreciation_name] = qdepreciation_data
        QtrDataFrame[q_interest_name] = qinterest_data
        QtrDataFrame[q_pbt_name] = qpbt_data
        QtrDataFrame[q_tax_name] = qtax_data
        QtrDataFrame[q_net_profit_name] = qnet_profit_data
        QtrDataFrame[q_op_profit_name] = qope_profit_data

        self.set_q_sales_data(QtrDataFrame)

        BalanceSheetDataFrame = pd.DataFrame()
        BalanceSheetDataFrame[bal_report_date_name] = dates #bal_report_date
        BalanceSheetDataFrame[bal_equity_share_capital_name] = bal_equity_share_capital
        BalanceSheetDataFrame[bal_reserves_name] = bal_reserves
        BalanceSheetDataFrame[bal_borrowings_name] = bal_borrowings
        BalanceSheetDataFrame[bal_other_liabilities_name] = bal_other_liabilities
        BalanceSheetDataFrame[bal_tot_liabilities_name] = bal_tot_liabilities
        BalanceSheetDataFrame[bal_net_block_name] = bal_net_block
        BalanceSheetDataFrame[bal_cap_wip_name] = bal_cap_wip
        BalanceSheetDataFrame[bal_investments_name] = bal_investments
        BalanceSheetDataFrame[bal_other_assets_name] = bal_other_assets
        BalanceSheetDataFrame[bal_tot_assets_name] = bal_tot_assets
        BalanceSheetDataFrame[bal_receivable_name] = bal_receivable
        BalanceSheetDataFrame[bal_inventory_name] = bal_inventory
        BalanceSheetDataFrame[bal_cash_in_bank_name] = bal_cash_in_bank
        BalanceSheetDataFrame[bal_no_of_equity_shares_name] = bal_no_of_equity_shares
        BalanceSheetDataFrame[bal_new_bonus_shares_name] = bal_new_bonus_shares
        BalanceSheetDataFrame[bal_face_value_name] = bal_face_value

        self.set_balance_sheet_data(BalanceSheetDataFrame)

        CashFlowDataFrame = pd.DataFrame()
        CashFlowDataFrame[cf_report_date_name] = dates
        CashFlowDataFrame[cf_cash_from_operating_activity_name] = cf_cash_from_operating_activity
        CashFlowDataFrame[cf_cash_from_investing_activity_name] = cf_cash_from_investing_activity
        CashFlowDataFrame[cf_cash_from_financial_activity_name] = cf_from_financing_activity
        CashFlowDataFrame[cf_net_cash_flow_name] = cf_net_cash_flow

        self.set_cash_flow_data(CashFlowDataFrame)

        #st.write(qdates)
        #st.write(QtrDataFrame)



    def qualitative_analysis(self):
        '''
        TODO:
        1. Management Background: who they are , experience, track record, any criminal record, etc..
        2. Business ethics: managment involved in scams, bribes, unfair business practices, etc..
        3. Corporate Governance: Board of Directors, Board of Advisors, etc..
        4. Business Model: How they make money, what is their product, how they sell, etc..
        5. Moat: What is their competitive advantage, how they are different from others, etc..
        6. Competition: Who are their competitors, what is their market share, etc..
        7. Industry: What is the industry outlook, what are the growth prospects, etc..
        8. Risks: What are the risks involved, what is the risk mitigation plan, etc..
        9. Financials: What are the financials, how they are performing, etc..
        10. Valuation: What is the valuation, how much it is worth, etc..
        11. Future Prospects: What are the future prospects, etc..
        12. Conclusion: What is the conclusion, etc..
        13. References: What are the references, etc..
        14. Minority Shareholders: How they are treated, etc..
        15. Share transactions: management buying/selling shares of company through promoter group
        16. related party transactions: is company tendering financial favors to known entities such as relatives, friends, etc..
        17. Salaries paid to promoters: are they taking high salaries, etc.. unsually a percentage of profits
        18. Operator activity in stocks: when promotors buy or sells shares, does the stock price move in the same direction
        19. shareholders: who are significant shareholders, etc..
        20. Political affiliation: is the company affiliated to any political party, etc..
        21. Promoter lifestyle: how do they live, etc..lifestyle, lavish simple etc...
         

        '''
        pass

    def calculate_ebitda_margin(self):
        '''
        
        EBITDA = [Operating Revenues - Operating Expense]
        Operating Revenues = [Total Revenue - Other Income]
        Operating Expense = [Total Expense - Finance Cost - Depreciation & Amortization]
        EBIDTA Margin = EBITDA / [Total Revenue - Other Income]
        EBITDA Margin = EBITDA / Total Revenue #not considering other income

        '''
        ebitda_margin = []
        for i in range(len(self.sales_data['Report Date'])):
            op_revenue = self.sales_data['Sales'][i] - self.sales_data['Other Income'][i]
            op_expense = (self.sales_data['Raw Material Cost'][i]+self.sales_data['Power and Fuel'][i]+self.sales_data['Other Mfr. Exp'][i]+self.sales_data['Employee Cost'][i]+self.sales_data['Selling and admin'][i]+self.sales_data['Other Expenses'][i]) - (self.sales_data['Depreciation'][i]+self.sales_data['Interest'][i])
            ebitda = op_revenue - op_expense
            ebidta_margin = ebitda/(self.sales_data['Sales'][i]-self.sales_data['Other Income'][i]) 
            ebitda_margin.append(ebidta_margin*100)
       
        if len(ebitda_margin) > 0:
            self.sales_data['EBITDA Margin'] = ebitda_margin

        print(self.sales_data['EBITDA Margin'])
       

    def quantitative_analysis(self):
        '''
        Financial Analysis:
        1. Profitability Ratios: Measure  how profitable company is
        2. Leverage Ratios: Is company able to do with its operations and obligations. 
        3. Valuation Ratios: Get sense of how cheap or expensive is stock. 
        4. Operating Ratios: Measure how well company is utilizing its assets and resources to generate profit.

        '''
        print("Inside quantitative_analysis")
        self.calculate_ebitda_margin()