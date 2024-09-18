import backtest_DA
import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(layout="wide")
file = st.file_uploader("Upload investment universe & price data", type=['xlsx', 'xls', 'csv'])
st.warning('Upload data.')

@st.cache_data
def load_data(file_path):

    df = pd.read_excel(file, sheet_name="data", parse_dates=["Date"], index_col=0, header=0).dropna()

    df2 = pd.read_excel(file, sheet_name="weight", parse_dates=["Date"], index_col=0, header=0).dropna()

    return df, df2

if file is not None:

    price, weight = load_data(file)

    price_list = list(map(str, price.columns))

    select = st.multiselect('Input Assets', price_list, price_list)
    input_list = price.columns[price.columns.isin(select)]
    input_price = price[input_list]

    if (st.button('Summit') or ('input_list' in st.session_state)):

        with st.expander('Portfolio', expanded=True):

            input_price = input_price.dropna()

            col40, col41, col42, col43, col44, col45, col46, col47, col48, col49 = st.columns([1, 1, 1, 1, 1, 0.3, 1, 1, 1, 1])

            with col40:

                start_date = st.date_input("Start", value=input_price.index[0])
                start_date = datetime.combine(start_date, datetime.min.time())

            with col41:

                end_date = st.date_input("End", value=input_price.index[-1])
                end_date = datetime.combine(end_date, datetime.min.time())

            with col42:

                freq_option = st.selectbox(
                    'Data Frequency', ('Daily', 'Monthly'))

            with col43:

                commission = st.number_input('Commission(%)')

            with col44:

                rebal = st.selectbox('Rebalancing', ('None','Daily', 'Monthly', 'Quarterly', 'Yearly'))
            ##############################################################################################################################
            with col46:

                st.session_state.bm1 = st.selectbox('Benchmark',['None'] + list(price.columns))
                #st.session_state.bm_price = price[st.session_state.bm1]

            with col47:

                if st.session_state.bm1 != 'None':

                    w1 = st.number_input('bm1 (%)', 0, 100, 60, 1)

            with col48:

                if st.session_state.bm1 == 'None':

                    st.session_state.bm_price = price
                    w1 = 0

                    st.session_state.bm2 = []

                if st.session_state.bm1 != 'None':

                    st.session_state.bm2 = st.selectbox('Benchmark', ['None']+list(price.columns[price.columns!=st.session_state.bm1]))

                    if st.session_state.bm2 == 'None':

                        st.session_state.bm_price = price[st.session_state.bm1].to_frame()

                        st.session_state.bm_price = st.session_state.bm_price[(input_price.index >= start_date)
                                                           & (input_price.index <= end_date)].dropna()

                        #
                        # st.session_state.bm_weight = pd.DataFrame(index=st.session_state.bm_price.index)


                    if st.session_state.bm2 != 'None':

                        st.session_state.bm_price = pd.concat([price[st.session_state.bm1], price[st.session_state.bm2]], axis=1)

                        st.session_state.bm_price = st.session_state.bm_price[(input_price.index >= start_date)
                                                           & (input_price.index <= end_date)].dropna()

            with col49:

                if  st.session_state.bm1 != 'None' and st.session_state.bm2 != 'None':

                    w2 = st.number_input('bm2 (%)', 0, 100, 100-w1, 1)

            ##############################################################################################################################


            if freq_option == 'Daily':
                daily = True
                monthly = False
                annualization = 365
                freq = 1

            if freq_option == 'Monthly':
                daily = False
                monthly = True
                annualization = 12
                freq = 2

            if daily == True:
                st.session_state.input_price = input_price[
                    (input_price.index >= start_date) & (input_price.index <= end_date)]

            if monthly == True:
                st.session_state.input_price = input_price[(input_price.index >= start_date)
                                                           & (input_price.index <= end_date)
                                                           & (input_price.index.is_month_end == True)].dropna()

            st.session_state.input_list = input_list
            st.session_state.input_price = pd.concat([st.session_state.input_price,
                                                      pd.DataFrame({'Cash': [100] * len(st.session_state.input_price)},
                                                                   index=st.session_state.input_price.index)], axis=1)


            # st.session_state.input_price = st.session_state.input_price[~st.session_state.input_price.index.duplicated()]

            # weight['Cash'] = 1 - weight[weight.index >= st.session_state.input_price.index[0]].sum(axis=1)
            weight['Cash'] = 1 - weight.sum(axis=1)
            weight = backtest_DA.rebalancing_set(st.session_state.input_price, weight, rebal)


##############################################################################################################################
            if st.session_state.bm1 == 'None':
                st.session_state.bm_price = st.session_state.bm_price.loc[st.session_state.input_price.index]
                st.session_state.bm_weight = pd.DataFrame(w1/100, index=weight.index, columns = st.session_state.bm_price.columns)

            elif st.session_state.bm1 != 'None' and st.session_state.bm2 != 'None':
                st.session_state.bm_price = st.session_state.bm_price.loc[st.session_state.input_price.index]

                st.session_state.bm_weight = pd.DataFrame(index=weight.index, columns = st.session_state.bm_price.columns)
                st.session_state.bm_weight[:] = pd.DataFrame([[w1/100,w2/100]])

            elif st.session_state.bm1 != 'None' and st.session_state.bm2 == 'None':
                st.session_state.bm_price = st.session_state.bm_price.loc[st.session_state.input_price.index]
                st.session_state.bm_weight = pd.DataFrame(w1/100, index=weight.index, columns = st.session_state.bm_price.columns)
##############################################################################################################################



            col1, col2, col_space, col3, col4= st.columns([2.5, 2.5, 0.3, 2, 2])

            with col1:

                st.write(" ")
                st.write("Input Data")
                st.dataframe(pd.DataFrame(st.session_state.input_price, index = st.session_state.input_price.index.date))

            with col2:

                st.write(" ")
                st.write("Input Allocation")
                # st.write(pd.DataFrame(weight, index=weight.index.date))
                weight = st.data_editor(pd.DataFrame(weight, index=weight.index.date), use_container_width=True)
                weight.index = pd.to_datetime(weight.index)

            with col3:
                if st.session_state.bm1 != 'None':
                    st.write(" ")
                    st.write("BM Data")
                    st.dataframe(pd.DataFrame(st.session_state.bm_price, index = st.session_state.bm_price.index.date))

            with col4:
                if st.session_state.bm1 != 'None':
                    st.write(" ")
                    st.write("BM Allocation")
                    # st.write(pd.DataFrame(st.session_state.bm_weight, index=weight.index.date))
                    st.session_state.bm_weight = st.data_editor(pd.DataFrame(st.session_state.bm_weight, index=weight.index.date), use_container_width=True)
                    st.session_state.bm_weight.index = pd.to_datetime(st.session_state.bm_weight.index)
        if st.button('Simulation'):
            #
            # st.session_state.slider = (slider * 0.01).tolist()

            st.session_state.weight = weight

            (st.session_state.portfolio_port, st.session_state.allocation_f,
             st.session_state.bm_portfolio_port,st.session_state.bm_allocation_f)  = \
                backtest_DA.simulation(st.session_state.input_price, st.session_state.weight,
                                       st.session_state.bm_price, st.session_state.bm_weight,
                                       commission, freq)

            # ''' bm 추가 '''
            # ##############################################################################################################################
            #
            # st.session_state.bm_portfolio_port, st.session_state.bm_float_weight = \
            #     backtest_DA.simulation(st.session_state.bm_price, st.session_state.bm_weight, commission, freq)
            #
            # ##############################################################################################################################

            st.session_state.alloc = st.session_state.allocation_f.copy()

            # st.session_state.ret = (st.session_state.input_price.iloc[1:] / st.session_state.input_price.shift(
            #     1).dropna()) - 1
            st.session_state.ret = st.session_state.portfolio_port.pct_change().fillna(0)

            # st.session_state.attribution = ((st.session_state.ret * (
            #     st.session_state.alloc.shift(1).dropna())).dropna() + 1).prod(axis=0) - 1

            st.session_state.attribution = (((st.session_state.alloc).mul(st.session_state.ret, axis=0)+1).prod(axis=0)-1)
            st.session_state.attribution =  (st.session_state.portfolio_port[-1] / 100-1)*st.session_state.attribution/st.session_state.attribution.sum()


            if monthly == True:
                st.session_state.portfolio_port = st.session_state.portfolio_port[
                    st.session_state.portfolio_port.index.is_month_end == True]

            st.session_state.drawdown = backtest_DA.drawdown(st.session_state.portfolio_port)

            st.session_state.bm_drawdown = backtest_DA.drawdown(st.session_state.bm_portfolio_port)

            st.session_state.input_price_N = st.session_state.input_price[
                (st.session_state.input_price.index >= st.session_state.portfolio_port.index[0]) &
                (st.session_state.input_price.index <= st.session_state.portfolio_port.index[-1])]
            st.session_state.input_price_N = 100 * st.session_state.input_price_N / st.session_state.input_price_N.iloc[0, :]


            st.session_state.portfolio_port.index = st.session_state.portfolio_port.index.date
            st.session_state.drawdown.index = st.session_state.drawdown.index.date
            st.session_state.input_price_N.index = st.session_state.input_price_N.index.date
            st.session_state.alloc.index = st.session_state.alloc.index.date

            st.session_state.bm_portfolio_port.index = st.session_state.bm_portfolio_port.index.date
            st.session_state.bm_drawdown.index = st.session_state.bm_drawdown.index.date

            st.session_state.result = pd.concat([st.session_state.portfolio_port,
                                                 st.session_state.drawdown,
                                                 st.session_state.input_price_N,
                                                 st.session_state.alloc],
                                                axis=1)

            st.session_state.START_DATE = st.session_state.portfolio_port.index[0].strftime("%Y-%m-%d")
            st.session_state.END_DATE = st.session_state.portfolio_port.index[-1].strftime("%Y-%m-%d")
            st.session_state.Total_RET = round(float(st.session_state.portfolio_port[-1] / 100-1)*100, 2)
            st.session_state.Anuuual_RET = round(float(((st.session_state.portfolio_port[-1] / 100) ** (
                    annualization / (len(st.session_state.portfolio_port) - 1)) - 1) * 100), 2)
            st.session_state.Anuuual_Vol = round(
                                            float(np.std(st.session_state.portfolio_port.pct_change().dropna())
                                                  * np.sqrt(annualization) * 100),2)

            st.session_state.MDD = round(float(min(st.session_state.drawdown) * 100), 2)
            st.session_state.Daily_RET = st.session_state.portfolio_port.pct_change().dropna()

        st.session_state.result_expander1 = st.expander('Result', expanded=True)

        with st.session_state.result_expander1:

            if 'weight' in st.session_state:


                st.write(" ")

                col50, col51, col52, col53, col54 = st.columns([1, 1, 1, 1, 1])

                with col50:
                    st.info("Period: " + str(st.session_state.START_DATE) + " ~ " + str(st.session_state.END_DATE))

                with col51:
                    st.info("Total Return: " + str(st.session_state.Total_RET) + "%")


                with col52:
                    st.info("Annual Return: " + str(st.session_state.Anuuual_RET) + "%")


                with col53:
                    st.info("Annual Volatility: " + str(st.session_state.Anuuual_Vol) + "%")

                with col54:

                    st.info("MDD: " + str(st.session_state.MDD) + "%")

                col21, col22, col23, col24 = st.columns([1.7, 1.7, 3.5, 3.5])

                with col21:
                    st.write('NAV')
                    st.dataframe(pd.concat([st.session_state.portfolio_port.round(2),st.session_state.bm_portfolio_port.round(2)], axis=1))

                    st.download_button(
                        label="Download",
                        data=st.session_state.result.to_csv(index=True),
                        mime='text/csv',
                        file_name='Result.csv')

                with col22:
                    st.write('MDD')
                    st.dataframe(pd.concat([st.session_state.drawdown.round(2),st.session_state.bm_drawdown.round(2).rename('Benchmark')], axis=1))

                with col23:
                    st.write('Price')
                    st.dataframe((st.session_state.input_price_N).
                                 astype('float64').round(2))

                with col24:
                    st.write('Weight')
                    st.dataframe(st.session_state.alloc.applymap('{:.2%}'.format))

                st.write(" ")

                col31, col32 = st.columns([1, 1])

                with col31:
                    #st.write("Net Asset Value")
                    # fig = px.line(st.session_state.portfolio_port.round(2))
                    # fig.update_xaxes(title_text='Time', showgrid=True)
                    # fig.update_yaxes(title_text='NAV', showgrid=True)
                    # fig.update_layout(showlegend=False)
                    # st.plotly_chart(fig)

                    # 데이터프레임으로 변환
                    df1 = st.session_state.portfolio_port.reset_index()
                    df1.columns = ['Time', 'Value']
                    df1['Series'] = 'Portfolio'

                    df2 = st.session_state.bm_portfolio_port.reset_index()
                    df2.columns = ['Time', 'Value']
                    df2['Series'] = 'Benchmark'

                    # Alpha 계산

                    # Alpha 계산
                    pct_change_df1 = df1['Value'].pct_change()
                    pct_change_df2 = df2['Value'].pct_change()


                    df3_values = ((pct_change_df1 - pct_change_df2 + 1).cumprod()-1).fillna(1)
                    df3 = pd.DataFrame({
                        'Time': df1['Time'],
                        'Value': df3_values,
                        'Series': 'Alpha'
                    })

                    # 두 데이터프레임을 결합
                    st.session_state.combined_df = pd.concat([df1, df2])

                    # px.line을 사용하여 시각화
                    fig = px.line(st.session_state.combined_df, x='Time', y='Value', color='Series', title='Net Asset Value')

                    # 음영 그래프 추가
                    fig.add_trace(
                        go.Scatter(
                            x=df3['Time'],
                            y=df3['Value'],
                            mode='lines',
                            name='Alpha',
                            fill='tozeroy',  # 음영 처리
                            fillcolor='rgba(0, 100, 80, 0.15)',  # 그래프의 투명도 설정
                            yaxis='y2',  # 우측 Y축을 기준으로 설정
                            line=dict(color='rgba(0, 100, 80, 0)')  # 선의 색상을 투명하게 설정 (테두리 없음)
                        )
                    )

                    # X축, Y축 및 레이아웃 설정
                    fig.update_layout(
                        yaxis=dict(title='NAV'),  # 왼쪽 Y축
                        yaxis2=dict(
                            #title='',
                            overlaying='y',
                            side='right',
                            showgrid=False  # 이 축의 그리드라인을 숨기려면 False로 설정
                        ),
                        legend=dict(
                            x=0,
                            y=0.9,
                            traceorder='normal',
                            bgcolor='rgba(255, 255, 255, 0)',
                            bordercolor='rgba(0, 0, 0, 0)',
                            borderwidth=0.5
                        ),
                        height=550
                    )

                    st.plotly_chart(fig)


                with col32:

                    #st.write("Maximum Drawdown")
                    # fig_MDD = px.line(st.session_state.drawdown)
                    # fig_MDD.update_xaxes(title_text='Time', showgrid=True)
                    # fig_MDD.update_yaxes(title_text='MDD', showgrid=True)
                    # fig_MDD.update_layout(showlegend=False)
                    # st.plotly_chart(fig_MDD)

                    df1_d = st.session_state.drawdown.reset_index()
                    df1_d.columns = ['Time', 'Value']
                    df1_d['Series'] = 'Drawdown'

                    df2_d = st.session_state.bm_drawdown.reset_index()
                    df2_d.columns = ['Time', 'Value']
                    df2_d['Series'] = 'Benchmark Drawdown'

                    # 두 데이터프레임을 결합
                    st.session_state.combined_df_d = pd.concat([df1_d, df2_d])

                    # px.line을 사용하여 시각화
                    fig_MDD = px.line(st.session_state.combined_df_d, x='Time', y='Value', color='Series', title='Maximum Drawdown')

                    # X축, Y축 및 레이아웃 설정
                    fig_MDD.update_xaxes(title_text='Time', showgrid=True)
                    fig_MDD.update_yaxes(title_text='MDD', showgrid=True)
                    fig_MDD.update_layout(height=550,
                        legend=dict(
                            x=0,  # x 위치 (0: 왼쪽, 1: 오른쪽, 0.5: 중앙)
                            y=0.9,  # y 위치 (0: 하단, 1: 상단, 0.5: 중앙)
                            traceorder='normal',
                            bgcolor='rgba(255, 255, 255, 0)',
                            bordercolor='rgba(0, 0, 0, 0)',
                            borderwidth=0.5
                        )
                    )

                    st.plotly_chart(fig_MDD)


                col61, col62 = st.columns([1, 1])

                with col61:

                    st.download_button(
                        label="Download",
                        data=pd.concat([st.session_state.portfolio_port.round(2),st.session_state.bm_portfolio_port.round(2)], axis=1).to_csv(index=True),
                        mime='text/csv',
                        file_name='Net Asset Value.csv')

                with col62:

                    st.download_button(
                        label="Download",
                        data=pd.concat([st.session_state.drawdown.round(2),st.session_state.bm_drawdown.round(2)], axis=1).to_csv(index=True),
                        mime='text/csv',
                        file_name='Maximum Drawdown.csv')

                st.write(" ")

                col_a, col_b= st.columns([1, 1])

                with col_a:

                    fig_bar = px.bar(x=st.session_state.attribution.index, y=st.session_state.attribution * 100, title='Attribution')
                    fig_bar.update_xaxes(title=None, showgrid=True)
                    #fig_bar.update_xaxes(title_text='', showticklabels=False)
                    fig_bar.update_yaxes(title_text='Attribution(%)', showgrid=True, )
                    fig_bar.update_layout(height=550)
                    st.plotly_chart(fig_bar)
                #
                # with col_b:
                #     st.write("Correlation Matrix")
                #     st.session_state.corr = st.session_state.input_price.pct_change().dropna().corr().round(2)
                #     fig_corr = px.imshow(st.session_state.corr, text_auto=True, aspect="auto")
                #     fig_corr.update_layout(width=820)
                #     st.plotly_chart(fig_corr)

                with col_b:


                    # fig_4, ax_4 = plt.subplots(figsize=(20,10))
                    # ax_4.stackplot(st.session_state.EF['EXP_RET']*100, (st.session_state.EF*100).drop(['EXP_RET', 'STDEV'], axis=1).T,
                    #                labels = Target_Weight.index, alpha = 0.4, edgecolors="face", linewidths=2)
                    #
                    # handles, labels = ax_4.get_legend_handles_labels()
                    # ax_4.legend(reversed(handles), reversed(labels),loc='lower left', fontsize=14)
                    # plt.xticks(fontsize=15)
                    # plt.yticks(fontsize=15)
                    # plt.xlabel('Return(%)', fontsize=15, labelpad=20)
                    # plt.ylabel('Weight(%)', fontsize=15, labelpad=15)
                    # ax_4.margins(x=0, y=0)
                    #
                    # st.pyplot(fig_4)
                    # st.write(st.session_state.EF.iloc[:,2:].columns)


                    # stock_num = 11

                    fig_WE = px.area(st.session_state.alloc,  y=st.session_state.alloc.columns, title='Weight')
                    # # 첫 11개 종목은 빨간색 계열로 설정 (톤을 조금씩 구분)
                    # for i in range(stock_num):
                    #     # 빨간색 계열에서 점차 밝아지도록 설정
                    #     fig_WE.data[i].update(line=dict(color=f'rgba(255, {int(50 + (205 / 11) * i)}, {int(50 + (205 / 11) * i)}, 1)'))

                    # # 나머지 종목은 파란색 계열로 설정 (톤을 조금씩 구분)
                    # for i in range(stock_num, len(fig_WE.data)):
                    #     # 파란색 계열에서 점차 밝아지도록 설정
                    #     fig_WE.data[i].update(line=dict(
                    #         color=f'rgba({int(50 + (205 / (len(fig_WE.data) - 11)) * (i - 11))}, {int(50 + (205 / (len(fig_WE.data) - 11)) * (i - 11))}, 255, 1)'))

                    fig_WE.update_traces(line=dict(width=0.1))
                    fig_WE.update_layout(
                        legend=dict(
                            x=0.0,
                            y=-0.0,
                            traceorder='normal',
                            bgcolor='rgba(255, 255, 255, 0.5)',
                            bordercolor='rgba(0, 0, 0, 0)',
                            borderwidth=0.01
                        )
                    )

                    fig_WE.update_xaxes(title_text='Time', showgrid=True)
                    fig_WE.update_yaxes(title_text='Weight(%)', showgrid=True)
                    fig_WE.update_layout(height=550, legend=dict(x=0.8, font=dict(size=8)))

                    # # fig_WE.update_layout(height=500)
                    # fig_WE.add_vline(x=st.session_state.EF.loc[Target_index]["EXP_RET"], line_color="red", annotation_text="Target",
                    #                  annotation_position="top")

                    st.plotly_chart(fig_WE)

                    #st.write(st.session_state.ret * st.session_state.alloc)


                col71, col72 = st.columns([1, 1])

                with col71:

                    st.download_button(
                        label="Download",
                        data=(st.session_state.attribution).to_csv(index=True),
                        mime='text/csv',
                        file_name='Attribution.csv')

                with col72:

                    st.download_button(
                        label="Download",
                        data=st.session_state.allocation_f.to_csv(index=True),
                        mime='text/csv',
                        file_name='weight.csv')
                #
                # col73, col74 = st.columns([1, 1])

                # with col73:
                #     # weight 시계열 데이터프레임에서 중복된 행 제거
                #     weight_unique = st.session_state.weight.drop_duplicates()
                #
                #     # 첫 번째 고유 행만 사용
                #     weight_row = weight_unique.iloc[5]
                #     df_weight_unique = weight_row.reset_index()
                #
                #     # 열 이름을 지정
                #     df_weight_unique.columns = ['Asset', 'Value']  # 'Asset'은 자산명, 'Value'는 비중
                #
                #     df_weight_unique['Value'] = df_weight_unique['Value'].apply(lambda x: max(x, 0.0000001))
                #
                #     # 'stock'과 'bond'로 분류 (처음 11개는 'stock', 나머지는 'bond')
                #     df_weight_unique['Category'] = ['stock' if i < 11 else 'ficc' for i in range(len(df_weight_unique))]
                #
                #     # 트리맵 생성
                #     fig = px.treemap(df_weight_unique,
                #                      path=['Category', 'Asset'],  # 대분류와 자산명을 계층 경로로 사용
                #                      values='Value',  # 값으로 비중 사용
                #                      color='Value',  # 색상으로 비중 표시
                #                      color_continuous_scale='RdBu',  # 색상 스케일
                #                      )
                #
                #     # 제목을 'Portfolio'로 설정
                #     fig.update_layout(title='Portfolio')
                #
                #     # Hover 템플릿 수정 (비중을 정수로 표시)
                #     fig.update_traces(
                #         hovertemplate='<b>%{label}</b><br>Value: %{value:.0f}<extra></extra>'
                #     )
                #
                #     # Streamlit에서 트리맵 표시
                #     st.plotly_chart(fig)
                #
                # with col74:
                #     # Benchmark weight 데이터프레임에서 중복된 행 제거
                #     bm_weight_unique = st.session_state.bm_weight.drop_duplicates().iloc[0]
                #
                #     # 데이터프레임을 사용하기 위해 인덱스를 리셋하고 열 이름 지정
                #     df_bm_weight_unique = bm_weight_unique.reset_index()
                #     df_bm_weight_unique.columns = ['Asset', 'Value']  # 'Asset'은 자산명, 'Value'는 비중
                #
                #     # 'stock'과 'bond'로 분류 (각 자산을 stock과 bond로 분류)
                #     df_bm_weight_unique['Category'] = ['stock', 'bond']
                #
                #     # 트리맵 생성
                #     fig_bm = px.treemap(df_bm_weight_unique,
                #                         path=['Category', 'Asset'],  # 대분류와 자산명을 계층 경로로 사용
                #                         values='Value',  # 값으로 비중 사용
                #                         color='Value',  # 색상으로 비중 표시
                #                         color_continuous_scale='RdBu',  # 색상 스케일
                #                         )
                #
                #     # 제목을 'Benchmark'로 설정
                #     fig_bm.update_layout(title='Benchmark')
                #
                #     # Hover 템플릿 수정 (비중을 정수로 표시)
                #     fig_bm.update_traces(
                #         hovertemplate='<b>%{label}</b><br>Value: %{value:.0f}<extra></extra>'
                #     )
                #
                #     # Streamlit에서 트리맵 표시
                #     st.plotly_chart(fig_bm)
                #
                # st.write(st.session_state.weight)
                # st.write(st.session_state.bm_weight)
                #
                # st.write(df_weight_unique)
                # st.write(df_bm_weight_unique)
                #
                # st.write(weight_row)
                #
                #
                # if df_weight_unique['Value'].sum() > 0:
                #     st.write("Portfolio 데이터 존재")
                # else:
                #     st.warning("Portfolio 데이터가 비어 있습니다.")
                #
                # if df_bm_weight_unique['Value'].sum() > 0:
                #     st.write("Benchmark 데이터 존재")
                # else:
                #     st.warning("Benchmark 데이터가 비어 있습니다.")


