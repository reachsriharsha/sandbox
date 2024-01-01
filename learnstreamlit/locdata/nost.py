import pandas as pd

def read_loc_data():
    eci_data = pd.read_csv('query_eci_results.csv',low_memory=False)
    print(type(eci_data))
    return eci_data

def main():
    loc_df = read_loc_data()
    print(loc_df.describe())
    print(loc_df['gtpu_access_tunnel_ip'].value_counts())
    loc_df['gtpu_access_tunnel_ip'].value_counts().to_csv('gtpu_access_tunnel_ip.csv')


if __name__ == "__main__":
    main()