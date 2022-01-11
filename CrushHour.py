"""
Loading all the libraries needed to run the script
"""
# Basic data reading and processing
import json
import os
import pandas as pd
import numpy as np

# Natural language processing packages
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from wordcloud import WordCloud
import string
import advertools as ad
import html

# Plotting and dataviz packages
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
prop = FontProperties(fname='/System/Library/Fonts/Apple Color Emoji.ttc')
plt.rcParams['font.family'] = prop.get_family()
import plotly.express as px
from plotly.subplots import make_subplots

# Creating a color palette to customize the charts color
color_palette = ["#fe3c72", "#FD267A", "#F54C27", "#FF7854", "#F8A84C", "#303742", "#667180"]

# Streamlit
import streamlit as st

# Create a function to read raw data and return dataframes
def read_data(file):
    usage = pd.DataFrame(list(file["Usage"]["app_opens"].keys()))
    for x in list(file["Usage"].keys()):
        usage[x] = list(file["Usage"][x].values())
        #usage = usage.drop(axis=1, columns=["advertising_id", "idfa"])
        usage.rename(columns={0: "date"}, inplace=True)
    messages = pd.json_normalize(file["Messages"], record_path="messages")
    messages = messages.drop(axis=1, columns=["type", "fixed_height"])
    return usage, messages

# Continue to clean the `usage` dataframe
def swipe_classifier(variable):
    if variable == "swipes_passes":
        return "Swiped Left"
    else:
        return "Swiped Right"
def match_result(variable):
    if variable == "non_matched":
        return "Liked and not matched"
    elif variable == "matches":
        return "Liked and matched"
    else:
        return "Disliked"

def df_usage():
    annual_usage = usage[["date", "swipes_likes", "swipes_passes", "superlikes", "matches"]]
    annual_usage["date"] = pd.to_datetime(annual_usage["date"]).dt.year
    annual_usage["non_matched"] = annual_usage["swipes_likes"] + annual_usage["superlikes"] - annual_usage["matches"]
    annual_usage = annual_usage[["date", "swipes_passes", "matches", "non_matched"]]
    annual_usage = annual_usage.melt(id_vars="date")
    annual_usage["swipe_type"] = annual_usage["variable"].apply(swipe_classifier)
    annual_usage["match_result"] = annual_usage["variable"].apply(match_result)
    return annual_usage

# Continue to clean the `messages` dataframe
# Create a function to tokenize the textual data
def tokenize(text):
    text = html.unescape(text) # Get rid of HTML symbols
    words = word_tokenize(text) # Tokenize the text
    words = [w.lower() for w in words] # Lower case for each word
    punctuations = str.maketrans('', '', string.punctuation) # Remove punctuations
    words = [w.translate(punctuations) for w in words]
    words = [word for word in words if word.isalpha()] # remove unalphabetical elements
    return words
# Create a function to count the number of words in a message
def len_count(message):
    return len(message)

def df_matches():
    messages['clean_message'] = messages['message'].apply(tokenize)
    messages['len_message'] = messages['clean_message'].apply(len_count)

    matches = messages[['to','sent_date','clean_message','len_message']]
    matches['sent_date'] = pd.to_datetime(matches['sent_date'])
    matches['last_sent_date'] = matches['sent_date']
    matches = matches.rename(columns={'sent_date':'first_sent_date', 'clean_message':'count_msg'})
    matches = matches.groupby('to').agg({'first_sent_date':'min',
                                         'last_sent_date':'max',
                                         'count_msg':'count',
                                         'len_message':'sum'
                                        }).reset_index()
    matches['date_diff'] = (matches['last_sent_date']-matches['first_sent_date']).dt.days
    return matches

def df_msg_time():
    msg_time = messages[['sent_date','len_message']]
    msg_time['sent_date'] = pd.to_datetime(msg_time['sent_date'])
    msg_time['sent_hour'] = msg_time['sent_date'].dt.hour
    msg_time['sent_weekday'] = msg_time['sent_date'].dt.weekday
    msg_time['sent_day'] = msg_time['sent_date'].dt.day
    msg_time['sent_month'] = msg_time['sent_date'].dt.month
    return msg_time

def clean_text(text):
    text = html.unescape(text) # Get rid of HTML symbols
    words = word_tokenize(text) # Tokenize the text
    words = [w.lower() for w in words] # Lower case for each word
    punctuations = str.maketrans('', '', string.punctuation) # Remove punctuations
    words = [w.translate(punctuations) for w in words]
    words = [word for word in words if word.isalpha()] # remove unalphabetical elements
    fr_stopwords = stopwords.words('french')
    my_fr_stopwords = ['ça', 'cest', 'pas', 'jai', 'tas', 'alors', 'non', 'aussi', 'quoi', 'si', 'sinon', 'daccord',
                       'très', 'va', 'vais', 'comme', 'donc', 'coup']
    fr_stopwords.extend(my_fr_stopwords)
    stop_words = set(stopwords.words('english')) | set(fr_stopwords)
    words = [w for w in words if not w in stop_words]
    return words

# Combining all the messages into one corpus
def create_corpus():
    list_msg=list(messages['message'])
    corpus=' '.join(list_msg)
    cleaned = clean_text(corpus)
    new_corpus = ' '.join(cleaned)
    return list_msg, new_corpus

# Using the advertools library to extract informations of emojis used in the messages
def emoji_summary():
    emoji_summary = ad.extract_emoji(list_msg)
    top_emoji = emoji_summary['top_emoji']
    df_emoji = pd.DataFrame(top_emoji, columns=["emoji", "frequency"])
    return df_emoji

# Plotting with the plotly express package

# Creating two charts using app usage data
def plot_annual_usage(df):
    fig1 = px.sunburst(df,
                       path=["date", "swipe_type", "match_result"],
                       values="value",
                       color_discrete_sequence=color_palette)
    col1, col2 = st.columns([3, 1])
    col1.subheader(":calendar: Swipes & Matches : Annual Stats")
    col1.plotly_chart(fig1, use_container_width=True)
    col2.subheader("KEY INDICATORS")
    col2.write("From "+str(y1_usg)+" to "+str(y2_usg)+", you had swiped right on "+str(total_swipes)+" profiles with " +str(total_matches)+" matches.")
    col2.write(":cupid: Avg Swipe Right Rate: " + str(round((total_likes / total_swipes) * 100,2)) + "%")
    col2.write(":revolving_hearts: Avg Match Rate: "+str(round((total_matches/total_likes)*100,2))+"%")

def plot_daily_usage(df):
    fig2 = px.line(df,
                   x=df.columns[0],
                   y=df.columns[1:8],
                   color_discrete_sequence=color_palette).update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")])))
    st.subheader(":date: Swipes & Matches : Daily Stats")
    st.plotly_chart(fig2, use_container_width=True)

def plot_total_msg(df):
    fig3 = px.scatter(df,
                      x="messages_sent",
                      y='messages_received',
                      trendline='ols',
                      title='Number of messages sent and received by day',
                      color_discrete_sequence=color_palette)
    st.subheader(":inbox_tray: Messages Sent vs Received")
    st.plotly_chart(fig3, use_container_width=True)

# Creating some subplots on the number and the length of messages sent
def plot_match_hist(df):
    fig4 = make_subplots(rows=1, cols=3, subplot_titles=["Number of messages per match",
                                                         "Number of words per message",
                                                         "Correlation number/length"],)
    trace1 = px.histogram(x=df["count_msg"],
                          nbins=50,
                          color_discrete_sequence=color_palette)
    trace2 = px.histogram(x=df["len_message"],
                          nbins=30,
                          color_discrete_sequence=color_palette)
    trace3 = px.scatter(df,
                        x="count_msg",
                        y="len_message",
                        color_discrete_sequence=color_palette,
                        size_max=40,
                        hover_name="to",)
    fig4.append_trace(trace1['data'][0], 1, 1)
    fig4.append_trace(trace2['data'][0], 1, 2)
    fig4.append_trace(trace3['data'][0], 1, 3)
    st.subheader(":bar_chart: Number & Length of Messages")
    st.plotly_chart(fig4, use_container_width=True)
    return fig4

# Creating subplots of histograms to visualize the number of sent messages by different measure of time
def plot_msg_hist(df):
    fig6 = make_subplots(rows=2,
                         cols=2,
                         subplot_titles=["Number of sent messages by month",
                                         "Number of sent messages by day of month",
                                         "Number of sent messages by day of week",
                                         "Number of sent messages by hour"],
                         column_titles=[""],)
    subplot1 = px.histogram(df, x="sent_month",
                            color_discrete_sequence=color_palette,
                            category_orders={"sent_month": ["Jan", "Feb", "Mar", "May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]})
    subplot2 = px.histogram(df, x="sent_day",color_discrete_sequence=color_palette)
    subplot3 = px.histogram(df, x="sent_weekday",color_discrete_sequence=color_palette)
    subplot4 = px.histogram(df, x="sent_hour",color_discrete_sequence=color_palette)
    fig6.append_trace(subplot1['data'][0], 1, 1)
    fig6.append_trace(subplot2['data'][0], 1, 2)
    fig6.append_trace(subplot3['data'][0], 2, 1)
    fig6.append_trace(subplot4['data'][0], 2, 2)
    st.subheader(":bar_chart: Number of Messages Sent by Periods of Time")
    st.plotly_chart(fig6, use_container_width=True)
    return fig6

# Creating a wordcloud
def plot_wordcloud(txt):
    st.subheader(":paperclip: Most Used Words in My Messages")
    st.write("According to the data you uploaded, these are the most frequently used words in messages you've sent.")
    num_words = st.selectbox("Select the maximum number of words to be displayed",[50, 100, 150, 200])
    wordcloud = WordCloud(width = 1400,
                          height = 900,
                          max_words=num_words,
                          background_color = "white").generate(txt)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot()
    return wordcloud

# Emoji Chart
def plot_emoji(df):
    fig8 = px.histogram(df, x = 'emoji', y = 'frequency', color_discrete_sequence=color_palette)
    st.subheader(':heart_eyes: My Favorite Emojis')
    st.write("According to the data you uploaded, these are the most frequently used emojis in messages you've sent.")
    st.plotly_chart(fig8, use_container_width=True)
    return fig8

def year_select(df, columnDate):
    df["year"] = pd.to_datetime(df[columnDate]).dt.year
    list_year = list(df["year"])
    time_span = (min(list_year), max(list_year))
    my_time_span = st.sidebar.slider("Time span of your " +
                                     [x for x in globals() if globals()[x] is df][0],
                                     min(list_year),
                                     max(list_year),
                                     time_span)
    first_year = my_time_span[0]
    last_year = my_time_span[1]
    df = df[(df['year'] >= first_year) & (df['year'] <= last_year )]
    return first_year, last_year, df

def year_summary(df):
    total_swipes = df["swipes_likes"].sum() + df["swipes_passes"].sum() + df["superlikes"].sum()
    total_likes = df["swipes_likes"].sum() + df["superlikes"].sum()
    total_matches = df["matches"].sum()
    return total_swipes, total_likes, total_matches

# Create an app using Streamlit
st.set_page_config(page_title="CrushHour - Your Personal Tinder Dashboard",
                   page_icon="🔥",
                   layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title(":fire: CrushHour")
st.markdown("Welcome to CrushHour, your personal Tinder dashboard")
st.sidebar.title(":eyes: About")
st.sidebar.markdown("Developed by Jiayue LIU")
st.sidebar.markdown(":mortar_board: MSc Data Management, PSB")
st.sidebar.markdown(":round_pushpin: Paris, France")
st.sidebar.markdown(":link: View source code on [GitHub](https://github.com/liu-jiayue/CrushHour)")
st.sidebar.title(":file_folder: File Uploader")
st.sidebar.markdown("You can request and download your Tinder data from [here](https://account.gotinder.com/data)")

# Create a file uploader so that users can upload their Tinder data
uploaded_file = st.sidebar.file_uploader("To begin with, please choose a dataset in .json format downloaded from Tinder")
if uploaded_file is not None:
    with st.sidebar.success('Your file has been uploaded successfully!'):
        raw_data = json.load(uploaded_file)
    with st.spinner('Please wait, your data is being processed...'):
        usage, messages = read_data(raw_data)
    st.sidebar.title(":white_check_mark: Data Selection")
    st.sidebar.markdown("Please select...")
    y1_usg, y2_usg, usage = year_select(usage, "date")
    total_swipes, total_likes, total_matches = year_summary(usage)
    y1_msg, y2_msg, messages = year_select(messages, "sent_date")
    annual_usage = df_usage()
    matches = df_matches()
    msg_time = df_msg_time()
    list_msg, new_corpus = create_corpus()
    df_emoji = emoji_summary()
    with st.spinner('Please wait, your data is being plotted...'):
        plot_annual_usage(annual_usage)
        plot_daily_usage(usage)
        plot_total_msg(usage)
        plot_match_hist(matches)
        plot_msg_hist(msg_time)
        plot_wordcloud(new_corpus)
        plot_emoji(df_emoji)
else:
    st.sidebar.write(":warning: File not uploaded yet.")