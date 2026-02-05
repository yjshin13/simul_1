import streamlit as st
import pandas as pd
import resampled_mvo
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")

file = st.file_uploader("Upload investment universe & price data", type=['xlsx', 'xls', 'csv'])
st.warning('Upload data.')

if file is not None:

    # price = pd.read_excel(file, sheet_name="price",
    #                    names=None, dtype={'Date': datetime}, index_col=0, header=0).dropna()
    price = pd.read_excel(file, sheet_name="price", parse_dates=["Date"], index_col=0, header=0).dropna()


    universe = pd.read_excel(file, sheet_name="universe",
                             names=None, dtype={'Date': datetime}, header=0)

    universe['key'] = universe['symbol'] + " - " + universe['name']

    select = st.multiselect('Input Assets', universe['key'], universe['key'])
    assets = universe['symbol'][universe['key'].isin(select)]

    input_price = price[list(assets)]
    input_universe = universe[universe['symbol'].isin(list(assets))].drop(['key'], axis=1)
    input_universe = input_universe.reset_index(drop=True) #index 깨지면 Optimization 배열 범위 초과 오류 발생

    with st.form("Resampling Parameters", clear_on_submit=False):

        st.subheader("Resampling Parameters:")

        col20, col21, col22, col23 = st.columns([1,1,1,3])

        with col20:

            start_date = st.date_input("Start", value = input_price.index[0])
            start_date = datetime.combine(start_date, datetime.min.time())

        with col21:

            end_date = st.date_input("End", value = input_price.index[-1])
            end_date = datetime.combine(end_date, datetime.min.time())

        with col22:


            if st.checkbox('Daily', value=False):

                daily = True
                monthly = False
                annualization = 252
                freq = "daily"

            if st.checkbox('Monthly', value=True):

                daily = False
                monthly = True
                annualization = 12
                freq = "monthly"


        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            Growth_range = st.slider('Equity Weight Constraint', 0, 100, (0, 40), 1)
            nPort = st.number_input('Efficient Frontier Points', value=200)

        with col2:
            Inflation_range = st.slider('Inflation Weight Constraint', 0, 100, (0, 30), 1)
            nSim = st.number_input('Number of Simulations', value=200)

        with col3:
            Fixed_Income_range = st.slider('Fixed_Income Weight Constraint', 0, 100, (50, 100), 1)
            Target = st.number_input('Select Target Return(%)', value=4.00)

            constraint_range = [Growth_range,Inflation_range,Fixed_Income_range]

        summit = st.form_submit_button("Summit")



        if summit and (('EF' not in st.session_state) or ([st.session_state.nPort, st.session_state.nSim,
                       st.session_state.constraint_range, list(st.session_state.input_price.columns)] \
                       != [nPort, nSim, constraint_range, list(input_price.columns)])):

            if daily==True:

                st.session_state.input_price = input_price[(input_price.index>=start_date) & (input_price.index<=end_date)]

            if monthly==True:

                st.session_state.input_price = input_price[(input_price.index>=start_date)
                                                           & (input_price.index<=end_date)
                                                           & (input_price.index.is_month_end==True)]

            st.session_state.input_universe = input_universe
            st.session_state.nPort = nPort
            st.session_state.nSim = nSim
            st.session_state.constraint_range = constraint_range

            st.session_state.EF = resampled_mvo.simulation(st.session_state.input_price,
                                                           st.session_state.nSim, st.session_state.nPort,
                                                           st.session_state.input_universe,
                                                           st.session_state.constraint_range,
                                                           annualization)
            A = st.session_state.input_universe.copy()
            A.index = st.session_state.input_universe['symbol']
            Result = pd.concat([A.drop(['symbol'], axis=1).T, st.session_state.EF.applymap('{:.6%}'.format)], axis=0, join='outer')
            new_col = Result.columns[-2:].to_list() + Result.columns[:-2].to_list()
            st.session_state.Result = Result[new_col]
            st.session_state.freq_input = freq


    if 'EF' in st.session_state:

        if daily == True:
            st.session_state.input_price = input_price[(input_price.index>=start_date) & (input_price.index<=end_date)]

        if monthly == True:
            st.session_state.input_price = input_price[(input_price.index >= start_date)
                                                       & (input_price.index <= end_date)
                                                       & (input_price.index.is_month_end == True)]

        with st.expander("Optimization (Target: " + str(Target) + "%, " + st.session_state.freq_input + ")", expanded=True) :

            Target_index = (st.session_state.EF['EXP_RET'] - Target / 100).abs().idxmin()

            col_x, col_y, col_z = st.columns([1, 1, 2])

            with col_x:

                st.info("Expected Return: " + str(round(st.session_state.EF.loc[Target_index]["EXP_RET"]*100,2)) + "%")

            with col_y:

                st.info("Expected Risk: " + str(round(st.session_state.EF.loc[Target_index]["STDEV"]*100,2))+"%")

            st.write("")

            Target_Weight = st.session_state.EF.loc[Target_index]\
                            .drop(["EXP_RET", "STDEV"])


            st.subheader("")
            st.empty()


            col_a, col_b = st.columns([1, 1])

            with col_a:

                st.write("Efficient Frontier")
                EF_point = plt.figure(figsize=(20, 10))

                Point = np.full(len(st.session_state.EF),0)
                Point[Target_index] =2

                plt.scatter(st.session_state.EF['STDEV']*100, (st.session_state.EF['EXP_RET']*100).T,
                            marker='o',
                            s=130,
                            c=Point,
                           # alpha=0.7,
                            cmap='Paired',
                            alpha = 1,
                            linewidths=2,
                            edgecolors='lightblue')
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)

                plt.xlabel('Expected Risk(%)', fontsize=15, labelpad=20)
                plt.ylabel('Expected Return(%)', fontsize=15, labelpad=20)

                st.pyplot(EF_point)

            with col_b:

                st.write("Weight")
                x = (Target_Weight*100).values.round(2)
                y = Target_Weight.index

                fig_bar, ax_bar = plt.subplots(figsize=(20,10.8))
                width = 0.75  # the width of the bars
                bar = ax_bar.barh(y, x, color="lightblue", height= 0.8, )

                for bars in bar:
                    width = bars.get_width()
                    posx = width + 0.5
                    posy = bars.get_y() + bars.get_height() * 0.5
                    ax_bar.text(posx, posy, '%.1f' % width, rotation=0, ha='left', va='center', fontsize=13)


                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.xlabel('Weight(%)', fontsize=15, labelpad=20)
                plt.ylabel('Assets', fontsize=15, labelpad=15)
                ax_bar.margins(x=0.04, y=0.01)

                st.pyplot(fig_bar)

            col_c, col_d = st.columns([1, 1])

            with col_c:
                st.write("Weight vs Return")
                fig_4, ax_4 = plt.subplots(figsize=(20,10))
                ax_4.stackplot(st.session_state.EF['EXP_RET']*100, (st.session_state.EF*100).drop(['EXP_RET', 'STDEV'], axis=1).T,
                               labels = Target_Weight.index, alpha = 0.4, edgecolors="face", linewidths=2)

                handles, labels = ax_4.get_legend_handles_labels()
                ax_4.legend(reversed(handles), reversed(labels),loc='lower left', fontsize=14)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.xlabel('Return(%)', fontsize=15, labelpad=20)
                plt.ylabel('Weight(%)', fontsize=15, labelpad=15)
                ax_4.margins(x=0, y=0)

                st.pyplot(fig_4)

            with col_d:

                st.write("Weight vs Volatility")
                fig_3, ax_3 = plt.subplots(figsize=(20,10))
                ax_3.stackplot(st.session_state.EF['STDEV']*100, (st.session_state.EF*100).drop(['EXP_RET', 'STDEV'], axis=1).T,
                               labels = Target_Weight.index, alpha = 0.4, edgecolors="face", linewidths=2)

                handles, labels = ax_3.get_legend_handles_labels()
                ax_3.legend(reversed(handles), reversed(labels),loc='lower left', fontsize=14)

                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.xlabel('Volatility(%)', fontsize=15, labelpad=20)
                plt.ylabel('Weight(%)', fontsize=15, labelpad=15)
                ax_3.margins(x=0, y=0)

                st.pyplot(fig_3)



        st.download_button(
                label="Efficient Frontier",
                data=st.session_state.Result.to_csv(index=False),
                mime='text/csv',
                file_name='Efficient Frontier.csv')

