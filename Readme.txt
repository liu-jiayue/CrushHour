CrushHour is a student project at Paris School of Business.

The Python script here contains codes that build up an application that reads and analyses automatically the data.json file provided by Tinder. You can request your personal data from Tinder by logging in here : https://account.gotinder.com/data

Using the Streamlit architecture, the application enables users to read the json file and breaks it into different data frames (mainly one for usage data and another one for messages sent by the user).

The main visualisations include :
- A sunburst chart that summarises the annual stats of swipes and matches
- A line chart that depicts daily usage data;
- A set of histograms that showcases the number of sent messages per match and number of words per sent message;
- A set of histograms that highlights the messaging habits over time
- A word cloud that shows the most frequently used words in all of the sent messages sent;
- A bar chart that plots the most used emojis used in all of the sent messages

Generally, with the sliders built in the sidebar, users can also select the time span of the data they would like to visualise on the dashboard.