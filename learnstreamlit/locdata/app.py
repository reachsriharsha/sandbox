import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk


def is_mos_acceptable(mos):
    #print(type(mos))
    if isinstance(mos, str):
        if "rtcp_mos" in mos:
            return 'blue'
        else:
            pass

    #print("value of mos is:", float(mos))
    if  float(mos) > 50:
        #print("===================doing greeen")
        return 'green'
    else:
        #print("doing red")
        return 'red'
        
    
chart_data = pd.DataFrame(
   np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
   columns=['latitude', 'longitude'])
eci_data = pd.read_csv('enb_114.csv',skipinitialspace=True)
#print(eci_data)
eci_data['color'] = eci_data['s_dl_rtcp_mos'].apply(is_mos_acceptable)

st.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        latitude=28.64,
        longitude=77.21,
        zoom=5,
        pitch=50,
    ),
    layers=[
        #pdk.Layer(
        #   'HexagonLayer',
        #   data=eci_data,
        #   get_position='[longitude, latitude, ]',
        #   radius=200,
        #   elevation_scale=4,
        #   elevation_range=[0, 1000],
        #   pickable=True,
        #   extruded=True,
        #),
        pdk.Layer(
            'ScatterplotLayer',
            data=eci_data,
            get_position='[longitude, latitude]',
            get_color='[200, 30, 0, 160]',
            get_radius=200,
            pickable=True,
            extruded=True,
        ),
        #pdk.Layer(
        #    "ColumnLayer",
        #    data=eci_data,
        #    get_elevation="s_dl_rtcp_mos",
        #    get_position='[longitude, latitude]',
        #    elevation_scale=1000,
        #    pickable=True,
        #    extruded=True,
        #    auto_highlight=True,
        #    radius=200,
        #    get_fill_color="color",
        #),
        pdk.Layer(
        'GridLayer',
        data=eci_data,
        get_position='[longitude, latitude]',
        get_color='[200, 30, 0, 160]',
        get_radius=200,
        pickable=True,
        extruded=True,
        cell_size=200, 
        elevation_scale=4
        ),
    ],
    tooltip={"text": "Cell Site:{cell_site}\nCell Name:{cell_name}\n DL MOS:{s_dl_rtcp_mos}"},
))