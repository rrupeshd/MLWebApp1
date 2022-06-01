import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

def Pro4():
    st.header('Credit Risk Assessment (Multi ML Model)')

    st.subheader(
        "This a Multi ML model for classifying credit risk of the customers.")

    df = pd.read_csv(r'./data/SouthGermanCredit.csv')
    dfL = pd.read_csv(r'./data/SouthGermanCreditwithlabels.csv')


    def load_data3(nrows):
        data = dfL
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis='columns', inplace=True)
        return data

    from sklearn.model_selection import train_test_split
    X = df.drop(['credit_risk'], axis=1)
    y = df["credit_risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)

    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # machine learning model_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import AdaBoostClassifier

    model_pipeline = []
    model_pipeline.append(LogisticRegression(solver='liblinear'))
    model_pipeline.append(SVC())
    model_pipeline.append(KNeighborsClassifier())
    model_pipeline.append(DecisionTreeClassifier())
    model_pipeline.append(RandomForestClassifier())
    model_pipeline.append(GaussianNB())
    model_pipeline.append(AdaBoostClassifier())

    from sklearn import metrics
    from sklearn.metrics import confusion_matrix

    model_list = ['Logistic Regression', 'SVM', 'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes', 'AdaBoost']
    acc_list = []
    auc_list = []
    cm_list = []

    for model in model_pipeline:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc_list.append(metrics.accuracy_score(y_test, y_pred))
        fpr, tpr, _thresholds = metrics.roc_curve(y_test, y_pred)
        auc_list.append(round(metrics.auc(fpr, tpr), 2))
        cm_list.append(confusion_matrix(y_test, y_pred))

    result_df = pd.DataFrame({'Model': model_list, 'Accuracy': acc_list, 'AUC': auc_list})
    result_df.sort_values('Accuracy', ascending=False)

    with st.sidebar:
        st.write("Select your choice.")
    selectmodel = st.sidebar.selectbox(
        'Select Model',
        (model_list)
    )

    if st.checkbox('Show me Model details', key="P4_1"):
        st.subheader('Accuracy')
        st.write('Accuracy is the most straightforward indicator of the model performance. It measure the percentage of accurate predictions: accuracy = (true positive + true negative) / (true positive + false positive + false negative + false positive)')
        score=float(result_df.Accuracy[result_df.Model==selectmodel])*100
        st.write(selectmodel + ' model accuracy score with feature scaling and PCA is : ' + str(score) + '%')
        score=float(result_df.AUC[result_df.Model==selectmodel])*100
        st.subheader('ROC & AUC')
        st.write('ROC is the plot of true positive rate against false positive rate at various classification threshold. AUC is the area under the ROC curve, and higher AUC indicates better model performance.')
        st.write(selectmodel + ' model AUC (Area Under the Curve) is : ' + str(score) + '%')
        st.subheader('Confusion matrix')
        st.write('Confusion matrix indicates the actual values vs. predicted values and summarize the true negative, false positive, false negative and true positive values in a matrix format.')
        fig1 = plt.figure(figsize=(10, 4))
        model = result_df.Model[result_df.Model==selectmodel]
        cm = cm_list[model_list.index(model.values)]
        ax = sns.heatmap(cm / np.sum(cm), annot=True,
                         fmt='.2%', cmap='Blues')
        ax.set_title('Confusion Matrix ')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Actual Values ')
        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels(['False', 'True'])
        ax.yaxis.set_ticklabels(['False', 'True'])
        ## Display the visualization of the Confusion Matrix.
        st.pyplot(fig1)



    if st.checkbox('Show raw data', key="P4_141"):
        st.subheader('Raw data')
        data = load_data3(1000)
        st.dataframe(data)
        st.write('Shape of dataset:', dfL.shape)


    if st.checkbox('Show All Model Accuracy', key="P4_142"):
        st.subheader('Results')
        data = result_df
        st.dataframe(data)
        st.write('Shape of dataset:', result_df.shape)

    if st.checkbox('EDA', key="43"):
        st.header("Univariate Analysis")
        fig = plt.figure(figsize=(15, 10))
        i = 0
        for column in df:
            sub = fig.add_subplot(5, 5, i + 1)
            sub.set_xlabel(column)
            df[column].plot(kind='hist')
            i = i + 1
        st.pyplot(fig)

        st.header('Categorical Features vs. Target — Grouped Bar Chart')
        # bar plot
        cat_list = ['status', 'credit_history', 'purpose', 'savings', 'employment_duration', 'installment_rate',
                    'personal_status_sex', 'other_debtors']
        fig = plt.figure(figsize=(24, 25))

        for i in range(len(cat_list)):
            column = cat_list[i]
            sub = fig.add_subplot(4, 2, i + 1)
            chart = sns.countplot(data=dfL, x=column, hue='credit_risk', palette='RdYlBu')
        st.pyplot(fig)

        st.header('Numerical Features vs. Target — Box Plot')
        # box plot
        num_list = ['duration', 'amount', 'age']
        fig = plt.figure(figsize=(12, 5))

        for i in range(len(num_list)):
            column = num_list[i]
            sub = fig.add_subplot(1, 3, i + 1)
            chart = sns.boxplot(data=dfL, y=column, x='credit_risk', palette='RdYlBu_r')
        st.pyplot(fig)

