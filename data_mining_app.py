import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
import numpy as np


data = {
    'Team': ['Arsenal', 'Chelsea', 'Liverpool', 'Man United', 'Man City', 'Tottenham', 'Newcastle', 'West Ham', 'Aston Villa', 'Leeds', 'Leicester', 'Everton', 'Wolves', 'Southampton', 'Norwich'],
    'Wins': [10, 7, 12, 8, 13, 6, 9, 5, 7, 4, 9, 8, 10, 6, 5],
    'Losses': [2, 4, 3, 5, 1, 4, 6, 5, 6, 7, 3, 4, 2, 5, 6],
    'Draws': [3, 5, 4, 4, 5, 4, 2, 3, 2, 3, 4, 6, 3, 3, 4],
    'Goals': [28, 22, 30, 20, 35, 18, 25, 15, 21, 17, 24, 23, 26, 19, 21],
    'Points': [33, 26, 40, 28, 44, 22, 29, 18, 23, 15, 31, 30, 33, 21, 19],
    'Home Wins': [6, 4, 7, 5, 9, 3, 5, 2, 4, 2, 5, 4, 6, 3, 3],
    'Away Wins': [4, 3, 5, 3, 4, 3, 4, 3, 3, 2, 4, 4, 4, 3, 2],
    'Goals Conceded': [18, 20, 15, 17, 12, 21, 19, 22, 16, 24, 18, 20, 14, 23, 25]
}

df = pd.DataFrame(data)

# Full-page Welcome Screen with colorful background
def display_welcome_screen():
    st.markdown(
        """
        <style>
        .reportview-container {
            background-color: #f0f0f5;
        }
        .sidebar .sidebar-content {
            background-color: #2b2b2b;
        }
        h1 {
            color: #4CAF50;
        }
        h2 {
            color: #2196F3;
        }
        p {
            font-size: 18px;
        }
        .welcome-container {
            text-align: center;
            background: linear-gradient(to right, #ff7e5f, #feb47b);
            padding: 150px 0;
            border-radius: 15px;
            color: white;
        }
        .welcome-container h2 {
            font-size: 36px;
            margin-bottom: 10px;
        }
        .welcome-container p {
            font-size: 20px;
        }
        .emoji-container {
            font-size: 40px;
            margin-top: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Welcome Screen with professional emojis
    st.markdown(
        """
        <div class="welcome-container">
            <h2>Welcome to the Premier League Data Dashboard! 🚀⚽</h2>
            <p>Explore and analyze data on your favorite Premier League teams 📊🎯</p>
            <p>Gain insights from multiple visualizations about team performance, goals, points, and more! 🏆✨</p>
            <p>Click the button below to enter the app and begin exploring the data! 🎉📉</p>
            <div class="emoji-container">
                🏅🔍📈📊📉🏆📅
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Wait for the user to click the Enter button
    if st.button("Enter App ➡️"):
        # Set a flag in session state to show the visualizations
        st.session_state.entered_app = True

# Display Visualizations after clicking Enter button
def display_visualizations():
    # App title
    st.title("Premier League Data Visualization 📊")

    # Display basic dataset information
    st.write("### These are the first rows and columns of your provided dataset:")
    st.write("This dataset contains information about Premier League teams, including their wins, losses, goals, points, home/away stats, and goals conceded.")
    st.write("The dataset will be used for various visualizations and analysis in this app. 📊")
    st.write(df.head())
    st.write("### Summary Statistics 📈")
    st.write(df.describe())

    # Visualization 1: Bar Chart for Team Performance (Wins vs. Goals)
    st.header("1. Bar Chart: Wins vs. Goals ⚖️")
    st.write("This bar chart compares the total **Wins** and **Goals** for each Premier League team. 🔄📊")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Team', y='Wins', color='skyblue', label='Wins')
    sns.barplot(data=df, x='Team', y='Goals', color='orange', label='Goals')
    plt.title("Comparison of Wins and Goals by Team")
    plt.xlabel("Teams")
    plt.ylabel("Count")
    plt.xticks(rotation=45) 
    plt.legend()
    st.pyplot(plt)

    # Visualization 2: Line Chart for Wins over Time (Team vs Wins)
    st.header("2. Line Chart: Team vs Wins 🏅")
    st.write("This line chart visualizes the number of **Wins** for each team across the season. 📅📈")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Team', y='Wins', marker='o', color='green')
    plt.title("Wins Over Time for Each Team")
    plt.xlabel("Teams")
    plt.ylabel("Wins")
    plt.xticks(rotation=45) 
    st.pyplot(plt)

    # Visualization 3: Scatter Plot for Goals vs. Points
    st.header("3. Scatter Plot: Goals vs. Points ⚽🏆")
    st.write("This scatter plot compares the **Goals** scored by each team against their **Points**. 📊📍")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Goals', y='Points', hue='Team', palette='tab10')
    plt.title("Goals vs. Points for Premier League Teams")
    plt.xlabel("Goals")
    plt.ylabel("Points")
    st.pyplot(plt)

    # Visualization 4: Histogram of Goals Scored by Teams
    st.header("4. Histogram: Distribution of Goals Scored 📉")
    st.write("This histogram shows the **distribution of goals** scored across teams. 📊🔢")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Goals'], bins=5, kde=True, color='purple')
    plt.title("Distribution of Goals Scored by Teams")
    plt.xlabel("Goals Scored")
    plt.ylabel("Frequency")
    st.pyplot(plt)

    # Visualization 5: Boxplot for Points Distribution
    st.header("5. Boxplot: Distribution of Points 📦")
    st.write("This boxplot shows the **distribution** of **Points** across teams, highlighting outliers. ⚖️")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Points', color='lightcoral')
    plt.title("Points Distribution Across Teams")
    plt.xlabel("Points")
    st.pyplot(plt)

    # Visualization 6: K-Means Clustering of Teams Based on Wins and Goals
    st.header("6. K-Means Clustering: Teams Based on Wins and Goals 🤖")
    st.write("In this visualization, we apply **K-Means clustering** to group teams based on their **Wins** and **Goals**. 🧠")
    kmeans = KMeans(n_clusters=3)
    df['Cluster'] = kmeans.fit_predict(df[['Wins', 'Goals']])

    # Plot the clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Wins', y='Goals', hue='Cluster', palette='Set1', s=100, marker='o')
    plt.title("K-Means Clustering: Teams Based on Wins and Goals")
    plt.xlabel("Wins")
    plt.ylabel("Goals")
    st.pyplot(plt)

    # Anomaly Detection with Isolation Forest
    st.header("7. Anomaly Detection using Isolation Forest 🚨")
    st.write("Anomaly detection helps to identify unusual patterns in the data that do not conform to expected behavior. Here, we apply **Isolation Forest** to detect anomalies in the dataset.")
    
    # Use Isolation Forest for anomaly detection
    isolation_forest = IsolationForest(contamination=0.1)  # Assuming 10% contamination
    df['Anomaly'] = isolation_forest.fit_predict(df[['Wins', 'Goals', 'Points']])

    # Mark anomalies in red
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Goals', y='Points', hue='Anomaly', palette='coolwarm', style='Anomaly', markers={-1: 'X', 1: 'o'})
    plt.title("Anomaly Detection: Goals vs Points")
    plt.xlabel("Goals")
    plt.ylabel("Points")
    st.pyplot(plt)

    # Random Forest Classification to predict 'High Performance'
    st.header("8. Random Forest Classification to Predict High Performance 🌲")
    st.write("Using **Random Forest**, we predict whether a team is **High Performance** based on its Wins, Goals, and Points.")

    # Define features and target
    df['High Performance'] = np.where((df['Wins'] > 8) & (df['Goals'] > 25), 1, 0)  # Target variable
    X = df[['Wins', 'Goals', 'Points']]
    y = df['High Performance']

    # Train Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Make predictions
    df['RF Prediction'] = rf.predict(X)

    st.write("Random Forest Predictions (1 = High Performance, 0 = Low Performance):")
    st.write(df[['Team', 'Wins', 'Goals', 'Points', 'RF Prediction']])

# Extra Charts Option Screen
def extra_charts_screen():
    st.write("### Choose a chart to visualize!")
    chart_options = ["Pie Chart", "Donut Chart", "Circular Plot", "Radar Chart", "Contour Plot"]
    chart_choice = st.selectbox("Select a chart type:", chart_options)

    if chart_choice == "Pie Chart":
        st.header("Pie Chart of Team Wins Distribution 🍰")
        st.write("This pie chart represents the distribution of **Wins** across the teams.")
        plt.figure(figsize=(8, 8))
        plt.pie(df['Wins'], labels=df['Team'], autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set3", len(df)))
        plt.title("Wins Distribution by Team")
        st.pyplot(plt)

    elif chart_choice == "Donut Chart":
        st.header("Donut Chart of Goals Scored 🎯")
        st.write("This donut chart shows the proportion of **Goals** scored by each team.")
        plt.figure(figsize=(8, 8))
        plt.pie(df['Goals'], labels=df['Team'], autopct='%1.1f%%', startangle=90, wedgeprops={'width': 0.3}, colors=sns.color_palette("Set2", len(df)))
        plt.title("Goals Scored by Team")
        st.pyplot(plt)

    elif chart_choice == "Circular Plot":
        st.header("Circular Plot of Points for Teams 🌐")
        st.write("This circular plot shows the **Points** of each team in a circular fashion.")
        plt.figure(figsize=(8, 8))
        angles = np.linspace(0, 2 * np.pi, len(df), endpoint=False).tolist()
        points = df['Points'].tolist()
        plt.subplot(111, polar=True)
        plt.bar(angles, points, width=0.3, bottom=0.0, color=sns.color_palette("Set1", len(df)))
        plt.title("Circular Plot of Points")
        st.pyplot(plt)

    # Radar Chart
    elif chart_choice == "Radar Chart":
        st.header("Radar Chart of Team Statistics 🦸‍♂️")
        st.write("This radar chart shows the **Wins**, **Goals**, and **Points** of each team.")
        
        # Create radar chart data
        teams = df['Team'].tolist()
        categories = ['Wins', 'Goals', 'Points']
        values = df[categories].values

        # Create a radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values = np.concatenate((values, values[:,[0]]), axis=1)  # Close the circle
        angles += angles[:1]  # Close the circle

        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, polar=True)
        for i in range(len(teams)):
            ax.plot(angles, values[i], linewidth=2, label=teams[i])

        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        plt.title("Radar Chart of Team Statistics")
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        st.pyplot(plt)

    # Contour Plot
    elif chart_choice == "Contour Plot":
        st.header("Contour Plot of Goals vs. Points 🌐")
        st.write("This contour plot shows the relationship between **Goals** and **Points**.")

        # Create a grid for contour plot
        x = np.linspace(df['Goals'].min(), df['Goals'].max(), 100)
        y = np.linspace(df['Points'].min(), df['Points'].max(), 100)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y)  # Example function for contour plot

        # Plot the contour
        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, Z, levels=20, cmap='coolwarm')
        plt.colorbar()
        plt.title("Contour Plot of Goals vs. Points")
        plt.xlabel("Goals")
        plt.ylabel("Points")
        st.pyplot(plt)

# Initialize session state variable for navigation
if 'entered_app' not in st.session_state:
    st.session_state.entered_app = False

if not st.session_state.entered_app:
    display_welcome_screen()
else:
    display_visualizations()
    extra_charts_screen()

