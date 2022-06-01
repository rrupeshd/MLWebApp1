import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


def Pro3():
    st.write(
        "This a ML model for classifying safety of car using Decision tree algorithm on demo dataset.")
    dfc = pd.read_csv(r'./data/car_evaluation.csv', header=None)
    col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    dfc.columns = col_names

    XDT = dfc.drop(['class'], axis=1)
    ydt = dfc['class']

    # split X and y into training and testing sets

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(XDT, ydt, test_size=0.33, random_state=42)
    X_train1 = X_train.copy(deep=True)
    X_test1 = X_test.copy(deep=True)

    buying_dict = {'vhigh': 1, 'high': 2, 'med': 3, 'low': 4}
    maint_dict = {'vhigh': 1, 'high': 2, 'med': 3, 'low': 4}
    doors_dict = {'2': 2, '3': 3, '4': 4, '5more': 5}
    persons_dict = {'2': 2, '4': 4, 'more': 5}
    lug_boot_dict = {'small': 1, 'big': 3, 'med': 2}
    safety_dict = {'low': 1, 'high': 3, 'med': 2}

    X_train1.buying = X_train1.buying.map(buying_dict)
    X_train1.maint = X_train1.maint.map(maint_dict)
    X_train1.doors = X_train1.doors.map(doors_dict)
    X_train1.persons = X_train1.persons.map(persons_dict)
    X_train1.lug_boot = X_train1.lug_boot.map(lug_boot_dict)
    X_train1.safety = X_train1.safety.map(safety_dict)

    X_test1.buying = X_test1.buying.map(buying_dict)
    X_test1.maint = X_test1.maint.map(maint_dict)
    X_test1.doors = X_test1.doors.map(doors_dict)
    X_test1.persons = X_test1.persons.map(persons_dict)
    X_test1.lug_boot = X_test1.lug_boot.map(lug_boot_dict)
    X_test1.safety = X_test1.safety.map(safety_dict)

    st.title('Car Safety (Decision Tree)')

    with st.sidebar:
        st.write("Select your choice.")

    buying = st.sidebar.selectbox(
        'Select Buying Category',
        (dfc.buying.unique())
    )
    maint = st.sidebar.selectbox(
        'Select Maintenance Category',
        (dfc.maint.unique())
    )
    doors = st.sidebar.selectbox(
        'Select Doors',
        (dfc.doors.unique())
    )

    persons = st.sidebar.selectbox(
        'Select Person Capacity',
        (dfc.persons.unique())
    )

    lug_boot = st.sidebar.selectbox(
        'Select Luggage size',
        (dfc.lug_boot.unique())
    )

    safety = st.sidebar.selectbox(
        'Select Safety class',
        (dfc.safety.unique())
    )


    list1 = []
    list1.append(buying)
    list1.append(maint)
    list1.append(doors)
    list1.append(persons)
    list1.append(lug_boot)
    list1.append(safety)
    dfw = pd.DataFrame([list1])
    col_names1 = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    dfw.columns = col_names1

    dfw.buying = dfw.buying.map(buying_dict)
    dfw.maint = dfw.maint.map(maint_dict)
    dfw.doors = dfw.doors.map(doors_dict)
    dfw.persons = dfw.persons.map(persons_dict)
    dfw.lug_boot = dfw.lug_boot.map(lug_boot_dict)
    dfw.safety = dfw.safety.map(safety_dict)

    # import DecisionTreeClassifier

    from sklearn.tree import DecisionTreeClassifier

    # instantiate the DecisionTreeClassifier model with criterion entropy
    clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)

    # fit the model
    clf_en.fit(X_train1, y_train)
    # Predict the Test set results with criterion entropy
    y_pred_en = clf_en.predict(X_test1)
    # Check accuracy score with criterion entropy
    from sklearn.metrics import accuracy_score

    st.write('Model accuracy score with criterion entropy: {0:0.4f}'.format(accuracy_score(y_test, y_pred_en)))

    prediction = clf_en.predict(dfw)
    st.subheader(f"Your Car's Safety evaluation is in {prediction} class.")

    def load_data3(nrows):
        data = dfc
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis='columns', inplace=True)
        return data


    if st.checkbox('Show raw data', key="P3_1"):
        st.subheader('Raw data')
        data = load_data3(1000)
        st.dataframe(data)
        st.write('Shape of dataset:', dfc.shape)

    if st.checkbox('Show Tree diagram', key="P3_2"):
        from sklearn import tree

        fig = plt.figure(figsize=(10, 4))
        ax1 = tree.plot_tree(clf_en.fit(X_train1, y_train))
        st.pyplot(fig)
