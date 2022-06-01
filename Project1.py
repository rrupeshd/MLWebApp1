

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px

def Pro1():
	st.write(
	    "This a ML model for predicting flight price using Multi Linear Regression on Kaggle dataset.")
	clean_df = pd.read_csv(r'./data/Clean_Dataset.csv')
	del clean_df["Unnamed: 0"]
	x_df = clean_df.copy(deep=True)
	x_df.drop(['flight', 'price'], axis=1, inplace=True)
	y_df = clean_df["price"].copy(deep=True)

	airline_dict = {'SpiceJet': 1, 'AirAsia': 2, 'Vistara': 3, 'GO_FIRST': 4, 'Indigo': 5, 'Air_India': 6}
	source_city_dict = {'Delhi': 1, 'Mumbai': 2, 'Bangalore': 3, 'Kolkata': 4, 'Hyderabad': 5, 'Chennai': 6}
	departure_time_dict = {'Evening': 4, 'Early_Morning': 1, 'Morning': 2, 'Afternoon': 3, 'Night': 5, 'Late_Night': 6}
	stops_dict = {'zero': 1, 'one': 2, 'two_or_more': 3}
	arrival_time_dict = {'Evening': 4, 'Early_Morning': 1, 'Morning': 2, 'Afternoon': 3, 'Night': 5, 'Late_Night': 6}
	destination_city_dict = {'Delhi': 1, 'Mumbai': 2, 'Bangalore': 3, 'Kolkata': 4, 'Hyderabad': 5, 'Chennai': 6}
	class_dict = {'Economy': 1, 'Business': 2}

	x_df.airline = x_df.airline.map(airline_dict)
	x_df.source_city = x_df.source_city.map(source_city_dict)
	x_df.departure_time = x_df.departure_time.map(departure_time_dict)
	x_df.stops = x_df.stops.map(stops_dict)
	x_df.arrival_time = x_df.arrival_time.map(arrival_time_dict)
	x_df.destination_city = x_df.destination_city.map(destination_city_dict)
	x_df["class"] = x_df["class"].map(class_dict)

	x_mapped_df = x_df.copy()
	x_mapped_df.drop(['duration'], axis=1, inplace=True)

	st.title('Flight Price Prediction')

	with st.sidebar:
		st.write("Select your choice.")

	airline_name = st.sidebar.selectbox(
	    'Select Airline',
	    (clean_df.airline.unique())
	)
	source_city_name = st.sidebar.selectbox(
	    'Select Source City',
	    (clean_df.source_city.unique())
	)
	destination_city_name = st.sidebar.selectbox(
	    'Select Destination City',
	    (clean_df.destination_city.unique())
	)

	departure_time_name = st.sidebar.selectbox(
	    'Select Departure Time',
	    (clean_df.departure_time.unique())
	)

	arrival_time_name = st.sidebar.selectbox(
	    'Select Arrival Time',
	    (clean_df.arrival_time.unique())
	)

	stops_name = st.sidebar.selectbox(
	    'Select No. of Stops',
	    (clean_df.stops.unique())
	)

	class_name = st.sidebar.selectbox(
	    'Select Class',
	    (clean_df["class"].unique())
	)

	Days = st.sidebar.slider('Days to travel', 1, 10, step=1)
	params = Days

	st.subheader(
	    f"You have selected {airline_name} airlines from {source_city_name} to {destination_city_name} in {departure_time_name} for {class_name} class.")


	# Splitting the dataset into the Training set and Test set
	from sklearn.model_selection import train_test_split
	X_train1, X_test1, y_train1, y_test1 = train_test_split(x_mapped_df, y_df, test_size=0.33, random_state=42)

	# Fitting Multiple Linear Regression to the Training set
	from sklearn.linear_model import LinearRegression
	model1 = LinearRegression()
	model1.fit(X_train1, y_train1)

	# Predicting the Test set results
	y_pred1 = model1.predict(X_test1)

	# Calculating the R squared value
	from sklearn.metrics import r2_score
	score=r2_score(y_test1, y_pred1)


	X_testlist=[airline_dict[airline_name], source_city_dict[source_city_name], departure_time_dict[departure_time_name], stops_dict[stops_name], arrival_time_dict[arrival_time_name], destination_city_dict[destination_city_name], class_dict[class_name],params]
	x_new_df=x_mapped_df[0:2].copy(deep=True)
	x_new_df.iloc[0] = X_testlist

	y_pred2 = model1.predict(x_new_df[0:1])
	predicated_price=str(y_pred2)[1:-1]
	predicated_price=float(predicated_price)
	st.title("Your estimated flight price is Rs. " + str(int(predicated_price)))


	st.write("R squared value is :" + str(score))

	df_selection = clean_df.query(
		"airline == @airline_name")


	if st.checkbox('Show raw data', key="P1_1"):
		st.subheader('Raw data of selected airlines')
		data = df_selection
		st.dataframe(data)
		st.write('Shape of dataset:', df_selection.shape)

	if st.checkbox('Show EDA', key="P1_2"):
		st.text("Simple EDA of raw data")

		bar_df=clean_df.airline.value_counts()
		st.bar_chart(bar_df)
		airline_avg=clean_df.groupby(['airline'])['price'].mean()
		airline_avg=airline_avg.to_frame()
		st.area_chart(airline_avg)

		st.header("Airline Price from source city")
		fig = plt.figure(figsize=(10, 4))
		ax=sns.countplot(x="airline", data=clean_df, order=clean_df.airline.value_counts().index)
		ax.bar_label(ax.containers[0])
		ax.set_xlabel("Top Airlines")
		ax.set_ylabel("Number of flights")
		st.pyplot(fig)

		rates_by_airline_line = (
			clean_df.groupby(by=['airline'])['price'].mean())
		fig_airline_sales = px.bar(
			rates_by_airline_line,
			x="price",
			y=rates_by_airline_line.index,
			orientation="h",
			title="<b>Rates_by_airline_line</b>",
			color_discrete_sequence=["#0083B8"] * len(rates_by_airline_line),
			template="plotly_white",
		)
		fig_airline_sales.update_layout(
			plot_bgcolor="rgba(0,0,0,0)",
			xaxis=(dict(showgrid=False))
		)
		left_column, right_column = st.columns(2)
		left_column.plotly_chart(fig_airline_sales, use_container_width=True)
		#right_column.plotly_chart(fig_product_sales, use_container_width=True)

