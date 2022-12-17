from typing import List

import menu as menu
import option as option
import streamlit.components.v1 as com
from PIL import Image
import pd as pd
import plost
import streamlit as st
from matplotlib.patches import ConnectionPatch
import graphviz
from streamlit_option_menu import option_menu

import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings



warnings.filterwarnings('ignore')

train_df = pd.read_csv("Dashboard/Titanic.csv")
traincopy_df = train_df.copy()
traincopy_df = traincopy_df.drop(['Name', 'Cabin', 'Ticket'], axis=1)
traincopy_df = traincopy_df.interpolate()

# Found 0.42 as minimum age, which is not possible. Therefore, rounded off the entire column
traincopy_df['Age'] = round(traincopy_df['Age'])
# Converting dataType from Float to Int
traincopy_df['Age'] = pd.to_numeric(traincopy_df['Age'], downcast='signed')

# New Table is getting formed
freq = [0] * 10
fmale = [0] * 10
ffemale = [0] * 10
i = 0

for x in traincopy_df['Age']:
    if x < 10:
        freq[0] += 1
        if traincopy_df['Sex'][i] == 'male':
            fmale[0] += 1
        else:
            ffemale[0] += 1
    elif x < 20:
        freq[1] += 1
        if traincopy_df['Sex'][i] == 'male':
            fmale[1] += 1
        else:
            ffemale[1] += 1
    elif x < 30:
        freq[2] += 1
        if traincopy_df['Sex'][i] == 'male':
            fmale[2] += 1
        else:
            ffemale[2] += 1
    elif x < 40:
        freq[3] += 1
        if traincopy_df['Sex'][i] == 'male':
            fmale[3] += 1
        else:
            ffemale[3] += 1
    elif x < 50:
        freq[4] += 1
        if traincopy_df['Sex'][i] == 'male':
            fmale[4] += 1
        else:
            ffemale[4] += 1
    elif x < 60:
        freq[5] += 1
        if traincopy_df['Sex'][i] == 'male':
            fmale[5] += 1
        else:
            ffemale[5] += 1
    elif x < 70:
        freq[6] += 1
        if traincopy_df['Sex'][i] == 'male':
            fmale[6] += 1
        else:
            ffemale[6] += 1
    elif x < 80:
        freq[7] += 1
        if traincopy_df['Sex'][i] == 'male':
            fmale[7] += 1
        else:
            ffemale[7] += 1
    elif x < 90:
        freq[8] += 1
        if traincopy_df['Sex'][i] == 'male':
            fmale[8] += 1
        else:
            ffemale[8] += 1
    else:
        freq[9] += 1
        if traincopy_df['Sex'][i] == 'male':
            fmale[9] += 1
        else:
            ffemale[9] += 1
    i += 1

AgeLevel = {'Age(In Years)': ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'],
            'Total People': [freq[0], freq[1], freq[2], freq[3], freq[4], freq[5], freq[6], freq[7], freq[8], freq[9]],
            'Male': [fmale[0], fmale[1], fmale[2], fmale[3], fmale[4], fmale[5], fmale[6], fmale[7], fmale[8],
                     fmale[9]],
            'Female': [ffemale[0], ffemale[1], ffemale[2], ffemale[3], ffemale[4], ffemale[5], ffemale[6], ffemale[7],
                       ffemale[8], ffemale[9]]}

# Total Male and Female
MALE = 0
FEMALE = 0

for x in AgeLevel['Male']:
    MALE += x
for y in AgeLevel['Female']:
    FEMALE += y

Male_Female = np.array([MALE, FEMALE])

survive = np.array(traincopy_df["Survived"])
ticketclass = np.array(traincopy_df["Pclass"])

# Indepth details of Males and Females from each port
# 0-S, 1-C, 2-Q
loc = [0] * 3
# Index(M,F): 0,1 for S; 2,3 for C; 4,5 for Q
loc_g = [0] * 6
# Counter for population
count = 0

# 0,1,2 for Males and 3,4,5 for Females
s = [0] * 6
c = [0] * 6
q = [0] * 6

for x in traincopy_df['Embarked']:
    if x == 'S':
        loc[0] += 1
        if traincopy_df['Sex'][count] == 'male':
            loc_g[0] += 1
            if ticketclass[count] == 1:
                s[0] += 1
            elif ticketclass[count] == 2:
                s[1] += 1
            else:
                s[2] += 1

        else:
            loc_g[1] += 1
            if ticketclass[count] == 1:
                s[3] += 1
            elif ticketclass[count] == 2:
                s[4] += 1
            else:
                s[5] += 1

    if x == 'C':
        loc[1] += 1
        if traincopy_df['Sex'][count] == 'male':
            loc_g[2] += 1
            if ticketclass[count] == 1:
                c[0] += 1
            elif ticketclass[count] == 2:
                c[1] += 1
            else:
                c[2] += 1
        else:
            loc_g[3] += 1
            if ticketclass[count] == 1:
                c[3] += 1
            elif ticketclass[count] == 2:
                c[4] += 1
            else:
                c[5] += 1
    if x == 'Q':
        loc[2] += 1
        if traincopy_df['Sex'][count] == 'male':
            loc_g[4] += 1
            if ticketclass[count] == 1:
                q[0] += 1
            elif ticketclass[count] == 2:
                q[1] += 1
            else:
                q[2] += 1
        else:
            loc_g[5] += 1
            if ticketclass[count] == 1:
                q[3] += 1
            elif ticketclass[count] == 2:
                q[4] += 1
            else:
                q[5] += 1
    count += 1

# Males-Females and Revenue Generation System Analysis
val = 890
i = 890
fd = 0
sd = 0
td = 0

fs = 0
ss = 0
ts = 0

# Male Dead
MD = [0] * 3
# Female Dead
FD = [0] * 3
# Male Survived
MS = [0] * 3
# Female Survived
FS = [0] * 3

# Revenue Generation
RevSM = [0] * 3
RevDM = [0] * 3
RevSF = [0] * 3
RevDF = [0] * 3

while (val > -1):
    if survive[val] == 0:
        if ticketclass[val] == 1:
            fd = fd + 1
            if traincopy_df['Sex'][i] == 'male':
                MD[0] += 1
                RevDM[0] += traincopy_df['Fare'][i]
            else:
                FD[0] += 1
                RevDF[0] += traincopy_df['Fare'][i]
        elif ticketclass[val] == 2:
            sd = sd + 1
            if traincopy_df['Sex'][i] == 'male':
                MD[1] += 1
                RevDM[1] += traincopy_df['Fare'][i]
            else:
                FD[1] += 1
                RevDF[1] += traincopy_df['Fare'][i]
        else:
            td = td + 1
            if traincopy_df['Sex'][i] == 'male':
                MD[2] += 1
                RevDM[2] += traincopy_df['Fare'][i]
            else:
                FD[2] += 1
                RevDF[2] += traincopy_df['Fare'][i]
    else:
        if ticketclass[val] == 1:
            fs = fs + 1
            if traincopy_df['Sex'][i] == 'male':
                MS[0] += 1
                RevSM[0] += traincopy_df['Fare'][i]
            else:
                FS[0] += 1
                RevSF[0] += traincopy_df['Fare'][i]
        elif ticketclass[val] == 2:
            ss = ss + 1
            if traincopy_df['Sex'][i] == 'male':
                MS[1] += 1
                RevSM[1] += traincopy_df['Fare'][i]
            else:
                FS[1] += 1
                RevSF[1] += traincopy_df['Fare'][i]
        else:
            ts = ts + 1
            if traincopy_df['Sex'][i] == 'male':
                MS[2] += 1
                RevSM[2] += traincopy_df['Fare'][i]
            else:
                FS[2] += 1
                RevSF[2] += traincopy_df['Fare'][i]
    val = val - 1
    i -= 1

survived = np.array([fd, sd, td])
dead = np.array([fs, ss, ts])

# Dashboard


st.set_page_config(
    page_title='Titanic Analysis'
)

# Heading of Dashboard

st.title("Titanic Analysis")
for x in range(2):  # Loop for blank Space
    st.write("#")  # For Blank Space


# General Page chart

def General():
    # BAR CHART

    for x in range(2):  # Loop for blank Space
        st.write("#")  # For Blank Space

    # Subheading
    st.subheader("Population Vs Age Group  ⚔️")
    st.write("#")  # For Blank Space

    # Bar Chart
    with st.container():
        # Create DataFrame
        AgeLevel_df = pd.DataFrame(
            {'Age(In Years)': ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90',
                               '90-100'],
             'Total People': [freq[0], freq[1], freq[2], freq[3], freq[4], freq[5], freq[6], freq[7], freq[8],
                              freq[9]], })
        st.bar_chart(AgeLevel_df, x='Age(In Years)', y='Total People', height=500, use_container_width=True)

        with st.expander("Insight"):
            st.write("""
                The graph above plainly illustrates that the **20-30 year** old age group was the most prevalent in Titanic.
            """)

    for x in range(2):  # Loop for blank Space
        st.write("#")  # For Blank Space

    # PIE CHART

    # Subheading
    st.subheader("🙎Male Vs 👩Female")
    st.write("#")  # For Blank Space

    # Pie Chart
    with st.container():
        pieChart = go.Figure(data=[go.Pie(labels=['Male', 'Female'], values=Male_Female, pull=[0.2, 0])])
        pieChart.update_layout(title_text='🙎 Vs 👩')
        st.plotly_chart(pieChart)

        with st.expander("Insight"):
            st.write("""
                The pie chart above plainly illustrates that the **Males** outnumber Females by a large margin.
            """)


def Location():
    # Location wise Analysis

    city = go.Figure(data=[go.Pie(labels=['Southampton', 'Cherbourg', 'Queenstown'], values=[loc[0], loc[1], loc[2]])])
    colors = ['#EC6B56', '#FFC154', '#47B39C']

    city.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=18,
                       marker=dict(colors=colors, line=dict(color='black', width=3)))

    city.update_layout(title_text='Passengers boarded (Males and Females)')
    st.plotly_chart(city)

    # Making Transition between Summary and Breakout
    st.markdown(
        """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 150px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    with st.container():
        col1, col2, col3, = st.columns(3, gap="large")
        with col2:
            st.metric(label="City-Wise Breakout Of Gender", value="⬇")

        t1, sc, t2, cq, t3 = st.tabs(
            ["Southampton City", "========================", "Cherbourg city", "========================", "Queenstown city"])

        with t1:
            # Southampton City
            with st.container():
                southampton = go.Figure(go.Sunburst(
                    labels=["Southampton", "3rd Class Ticket", "2nd Class Ticket", "1st Class Ticket", "Male(class-3)",
                            "Male(class-2)", "Female(class-3)", "Male(class-1)", "Female(class-2)", "Female(class-1)"],
                    parents=["", "Southampton", "Southampton", "Southampton", "3rd Class Ticket", "2nd Class Ticket",
                             "3rd Class Ticket", "1st Class Ticket", "2nd Class Ticket", "1st Class Ticket"],
                    values=[644, 353, 164, 127, 265, 97, 88, 79, 67, 48],
                    branchvalues="total",
                ))
                southampton.update_layout(margin=dict(t=0, l=0, r=0, b=0))
                st.plotly_chart(southampton)

        with sc:
            st.markdown(
                """
                <style>
                [data-testid="stMetricValue"] {
                    font-size: 150px;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            col1, col2, col3, = st.columns(3, gap="small")
            with col1:
                image = Image.open('Dashboard/Images/Southampton.jpeg')
                image = image.resize((500,500))
                st.image(image, caption='Southampton City')
            with col2:
                st.metric(label="", value="️<=>")
            with col3:
                image = Image.open('Dashboard/Images/Cherbourg.jpeg')
                image = image.resize((400,400))
                st.image(image, caption='Cherbourg City')

        with t2:
            # Cherbourg city
            with st.container():
                cherbourg = go.Figure(go.Sunburst(
                    labels=["Cherbourg", "1st Class Ticket", "3rd Class Ticket", "Female(class-1)", "Male(class-3)",
                            "Male(class-1)", "Female(class-3)", "2nd Class Ticket", "Male(class-2)", "Female(class-2)"],
                    parents=["", "Cherbourg", "Cherbourg", "1st Class Ticket", "3rd Class Ticket", "1st Class Ticket",
                             "3rd Class Ticket", "Cherbourg", "2nd Class Ticket", "2nd Class Ticket"
                             ],
                    values=[168, 85, 66, 43, 43, 42, 23, 17, 10, 7],
                    branchvalues="total",
                ))
                cherbourg.update_layout(margin=dict(t=0, l=0, r=0, b=0))
                st.plotly_chart(cherbourg)

        with cq:
            st.markdown(
                """
                <style>
                [data-testid="stMetricValue"] {
                    font-size: 150px;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            col1, col2, col3, = st.columns(3, gap="small")
            with col1:
                image = Image.open('Dashboard/Images/Cherbourg.jpeg')
                image = image.resize((500,500))
                st.image(image, caption='Cherbourg City')
            with col2:
                st.metric(label="", value="️<=>")
            with col3:
                image = Image.open('Dashboard/Images/Queenstown.jpeg')
                image = image.resize((400,400))
                st.image(image, caption='Queenstown City')

        with t3:
            # Queenstown city
            with st.container():
                queenstown = go.Figure(go.Sunburst(
                    labels=["Queenstown", "3rd Class Ticket", "Male(class-3)", "Female(class-3)", "2nd Class Ticket",
                            "Female(class-2)", "Male(class-2)", "1st Class Ticket", "Male(class-1)", "Female(class-1)"],
                    parents=["", "Queenstown", "3rd Class Ticket", "3rd Class Ticket", "Queenstown", "2nd Class Ticket",
                             "2nd Class Ticket", "Queenstown", "1st Class Ticket", "1st Class Ticket"],
                    values=[77, 72, 39, 33, 3, 2, 1, 2, 1, 1],
                    branchvalues="total",
                ))
                queenstown.update_layout(margin=dict(t=0, l=0, r=0, b=0))
                st.plotly_chart(queenstown)



def Survive():
    with st.container():
        # Combined Summary
        survive_summary = go.Figure(data=[
            go.Bar(name='Males', x=['1st class', '2nd class', '3rd class'], y=[MS[0], MS[1], MS[2]]),
            go.Bar(name='Females', x=['1st class', '2nd class', '3rd class'], y=[FS[0], FS[1], FS[2]])
        ])
        # Change the bar mode
        survive_summary.update_layout(barmode='group', title_text='Survival (Males VS Females)',
                                      xaxis_title='Ticket class',
                                      yaxis_title='Population')
        st.plotly_chart(survive_summary)

        # Making Transition between Summary and Breakout
        st.markdown(
            """
        <style>
        [data-testid="stMetricValue"] {
            font-size: 150px;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

    with st.container():
        col1, col2, col3, = st.columns(3, gap="large")
        with col2:
            st.metric(label="Male Vs Female Breakout", value="⬇")

    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            # Males Death
            males_survived = go.Figure(data=[go.Bar(x=['1st class', '2nd class', '3rd class'], y=[MS[0], MS[1], MS[2]],
                                                    hovertext=['41.28 %', '15.60 %', '43.12 %'])])
            males_survived.update_layout(title_text='Males Survival', xaxis_title='Ticket class',
                                         yaxis_title='Population', height=500, width=400)
            st.plotly_chart(males_survived)

        with c2:
            # Females Death
            females_survived = go.Figure(
                data=[go.Bar(x=['1st class', '2nd class', '3rd class'], y=[FS[0], FS[1], FS[2]],
                             hovertext=['39.05 %', '30.04 %', '30.91 %'])])
            females_survived.update_layout(title_text='Females Survival', xaxis_title='Ticket class',
                                           yaxis_title='Population', height=500, width=400)
            st.plotly_chart(females_survived)


def Death():
    with st.container():
        # Combined Summary
        dead_summary = go.Figure(data=[
            go.Bar(name='Males', x=['1st class', '2nd class', '3rd class'], y=[MD[0], MD[1], MD[2]]),
            go.Bar(name='Females', x=['1st class', '2nd class', '3rd class'], y=[FD[0], FD[1], FD[2]])
        ])
        # Change the bar mode
        dead_summary.update_layout(barmode='group', title_text='Death (Males VS Females)', xaxis_title='Ticket class',
                                   yaxis_title='Population')
        st.plotly_chart(dead_summary)

        with st.expander("Insight"):
            st.write("""
                The graph shows that males with third-class tickets died the most.
            """)


        # Making Transition between Summary and Breakout
        st.markdown(
            """
        <style>
        [data-testid="stMetricValue"] {
            font-size: 150px;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

    with st.container():
        col1, col2, col3, = st.columns(3, gap="large")
        with col2:
            st.metric(label="Male Vs Female Breakout", value="⬇")

    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            # Males Death
            males_dead = go.Figure(
                data=[go.Pie(labels=['1st class', '2nd class', '3rd class'], values=[MD[0], MD[1], MD[2]])])
            males_dead.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=18,
                                     marker=dict(colors=['#EC6B56', '#FFC154', '#47B39C'],
                                                 line=dict(color='black', width=3)))
            males_dead.update_layout(title_text='Males Death', height=500, width=378)
            st.plotly_chart(males_dead)
        with c2:
            # Females Death

            females_dead = go.Figure(
                data=[go.Pie(labels=['1st class', '2nd class', '3rd class'], values=[FD[0], FD[1], FD[2]])])
            females_dead.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=18,
                                       marker=dict(colors=['#FD6787', '#FFF44C', '#288EEB'],
                                                   line=dict(color='black', width=3)))
            females_dead.update_layout(title_text='Females Death', height=500, width=380)
            st.plotly_chart(females_dead)


def Revenue():
    with st.container():
        # Total Revenue
        revenue = go.Figure(data=[go.Pie(labels=['Males', 'Females'], text=['$', '$'],
                                         values=[sum(RevSM) + sum(RevDM), sum(RevSF) + sum(RevDF)])])

        revenue.update_traces(hoverinfo='label+percent', textinfo='label+text+value', textfont_size=18,
                              marker=dict(colors=['#FD6787', '#FFF44C'], line=dict(color='black', width=3)))

        revenue.update_layout(title_text='Total Revenue($)')
        st.plotly_chart(revenue)

        with st.expander("Insight"):
            st.write("""
                The graph demonstrates that the money generated by males and females is about equal.
            """)

    # Making Transition between Total Revenue And Summary
    st.markdown(
        """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 150px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    col1, col2, col3, = st.columns(3, gap="large")
    with col2:
        st.metric(label="Revenue Vs (Class+Survival)", value="⬇")

    with st.container():
        # Combined Summary of Survived+Dead Male and Female Revenue according to class
        revenue_summary = go.Figure(data=[
            go.Bar(name='Males Survived', x=['1st class', '2nd class', '3rd class'], y=[RevSM[0], RevSM[1], RevSM[2]]),
            go.Bar(name='Females Survived', x=['1st class', '2nd class', '3rd class'],
                   y=[RevSF[0], RevSF[1], RevSF[2]]),
            go.Bar(name='Males Dead', x=['1st class', '2nd class', '3rd class'], y=[RevDM[0], RevDM[1], RevDM[2]]),
            go.Bar(name='Females Dead', x=['1st class', '2nd class', '3rd class'], y=[RevDF[0], RevDF[1], RevDF[2]])
        ])
        # Change the bar mode
        revenue_summary.update_layout(barmode='group', title_text='Revenue ($)', xaxis_title='Ticket Type',
                                      yaxis_title='Revenue($)')
        st.plotly_chart(revenue_summary)

    with st.expander("Insight"):
        st.write("""
            The graph demonstrates that the female survivors with first -class tickets contributed the most to the revenue.
        """)

    # Making Transition between Summary And Gender Wise Breakout
    st.markdown(
        """
        <style>
        [data-testid="stMetricValue"] {
            font-size: 150px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3, = st.columns(3, gap="large")
    with col2:
        st.metric(label="Gender Wise Breakout", value="⬇")

    with st.container():
        # Revenue Breakout
        tab1, temporary, tab2 = st.tabs(
            ["Revenue from Male($)", "🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊", "Revenue from Female($)"])

        with tab1:
            # Revenue From Male
            revenue_male = go.Figure()
            revenue_male.add_trace(go.Bar(
                y=['Male'],
                x=[sum(RevSM)],
                name='Survived Male',
                orientation='h',
                marker=dict(
                    color='rgba(246, 78, 139, 0.6)',
                    line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                )
            ))
            revenue_male.add_trace(go.Bar(
                y=['Male'],
                x=[sum(RevDM)],
                name='Dead Male',
                orientation='h',
                marker=dict(
                    color='rgba(58, 71, 80, 0.6)',
                    line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                )
            ))

            revenue_male.update_layout(barmode='stack', title_text='Revenue from Male($)', xaxis_title='Revenue($)',
                                       yaxis_title='Gender',
                                       )
            st.plotly_chart(revenue_male)

        with temporary:
            st.markdown(
                """
                <style>
                [data-testid="stMetricValue"] {
                    font-size: 150px;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            col1, col2, col3, = st.columns(3, gap="large")
            with col1:
                st.metric(label="", value="👨")
            with col2:
                st.metric(label="", value="️↔")
            with col3:
                st.metric(label="", value="👩️")

        with tab2:
            revenue_female = go.Figure()
            revenue_female.add_trace(go.Bar(
                y=['Female'],
                x=[sum(RevSF)],
                name='Survived Female',
                orientation='h',
                marker=dict(
                    color='rgba(246, 78, 139, 0.6)',
                    line=dict(color='rgba(246, 78, 139, 1.0)', width=1)
                )
            ))
            revenue_female.add_trace(go.Bar(
                y=['Female'],
                x=[sum(RevDF)],
                name='Dead Female',
                orientation='h',
                marker=dict(
                    color='rgba(58, 71, 80, 0.6)',
                    line=dict(color='rgba(58, 71, 80, 1.0)', width=1)
                )
            ))

            revenue_female.update_layout(barmode="stack", title_text='Revenue from Female($)', xaxis_title='Revenue($)', )
            st.plotly_chart(revenue_female)


# Menu Bar
option = option_menu(
    menu_title=None,
    options=["General", "Location", "Survived", "Dead", "Revenue"],
    icons=["house", "pin-map-fill", "emoji-laughing", "emoji-frown", "currency-dollar"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#262730"},
        "icon": {"color": "#ffff33", "font-size": "26px"},
        "nav-link": {"font-size": "28px", "text-align": "center", "color": "white", "margin": "0px",
                     "--hover-color": "#FF4B4B"},
        "nav-link-selected": {"background-color": "#FF4B4B"},
    }
)

if option == "General":
    General()
elif option == "Location":
    Location()
elif option == "Survived":
    Survive()
elif option == "Dead":
    Death()
else:
    Revenue()

# Removing footer
st.markdown(
    """
    <style>
    .css-1rs6os.edgvbvh3
    {
        visibility: hidden;
    }
    .css-1lsmgbg.egzxvld0
    {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True
)
