import streamlit as st #streamlit for creating an interative dashboard
import warnings #warnings library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #libraries imported for data analysis
plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')
st.set_page_config(layout="wide")
st.title("Titanic Survival Prediction") #title of the dashboard
st.write("This is a machine learning model that predicts the survival of passengers on the RMS Titanic based on various factors like gender, age, class, fare, family on board, title, and embarked. The model is trained on the Titanic dataset and can be used to predict the survival of passengers in the test dataset. There are several passengers in the dataset, with some passengers' survival outcomes already documented, while others' survival outcomes haven't been documented.")

df=pd.read_csv("Titanic-Dataset.csv") #titanic dataset, predicted non documented survivors based on factors like gender, age, class, family on board, fare, title, and embarked.
#data cleaning like 0,1 changed to Yes/No 1,2,3 changed to First, Second, Third, Column Labels changed, S, C, Q replaced by full city names.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv') #Train-Test Split, 1-891 for training, 892-1309 for testing.

tab1, tab2, tab3 = st.tabs(["Data Overview", "Data Analysis", "Model Predictions"])
with tab1:
    st.header("Titanic Dataset")
    with st.expander("See Detailed Overview"):
        st.dataframe(df)
    st.write("The dataset contains information about **" + str(len(df)) + "** passengers, with **" + str(df['Survived'].eq('Yes').sum() + df['Survived'].eq('No').sum()) + "** survival outcomes documented and **" + str(df['Survived'].isnull().sum()) + "** survival outcomes not documented.")
with tab2:
    col1,col2,col3,col4= st.columns(4)
    with col1:
        st.header("Data Analysis")
        st.write("The data analysis section provides insights into the survival rates of passengers based on various factors.")
        st.subheader("Survival Analysis")
        f, ax = plt.subplots(1, 2, figsize=(12, 4)) 
        train['Survived'].value_counts().plot.pie( 
            explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=False) 
        ax[0].set_title('Percentage Survived') 
        #pie chart 
        sns.countplot(x='Survived', data=train, ax=ax[1])
        ax[1].bar_label(ax[1].containers[0])
        ax[1].set_ylabel('Quantity') 
        ax[1].set_title('Number of Survivors') #bar chart
        st.pyplot(f)
        with st.expander("See Insights"):
            st.write("The pie chart shows that only **" + str(round(train['Survived'].eq('Yes').mean() * 100, 1)) + "%** of passengers survived. The bar chart shows that there were **" + str(train['Survived'].eq('No').sum()) + "** non-survivors and **" + str(train['Survived'].eq('Yes').sum()) + "** survivors in the dataset.")
        st.subheader("Gender Analysis")
        f, ax = plt.subplots(1, 2, figsize=(12, 4)) 
        (train.assign(Survived= train['Survived'].eq('Yes'))
            [['Gender', 'Survived']]
            .groupby(['Gender']).mean()
            .plot.bar(ax=ax[0])) #gender comparison of survival rate and number of survivors by gender
        ax[0].set_title('Survival Rate by Gender') 
        ax[0].set_ylabel('Survival Rate')
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=0)
        sns.countplot(x='Gender', hue='Survived', data=train, ax=ax[1])
        ax[1].set_ylabel('Quantity') 
        ax[1].set_title('Number of Survivors by Gender') 
        st.pyplot(f)
        female_rate = train[train['Gender'] == 'Female']['Survived'].eq('Yes').mean()
        male_rate = train[train['Gender'] == 'Male']['Survived'].eq('Yes').mean()
        higher_gender = "Female" if female_rate > male_rate else "Male"
        with st.expander("See Insights"):
            st.write("The bar chart shows that the survival rate for females was **" + str(round(train[train['Gender'] == 'Female']['Survived'].eq('Yes').mean() * 100, 1)) + "%**, while the survival rate for males was **" + str(round(train[train['Gender'] == 'Male']['Survived'].eq('Yes').mean() * 100, 1)) + "%**. The count plot shows that there were **" + str(train[train['Gender'] == 'Female']['Survived'].eq('No').sum()) + "** female non-survivors and **" + str(train[train['Gender'] == 'Female']['Survived'].eq('Yes').sum()) + "** female survivors in the dataset. Meanwhile, there were **" + str(train[train['Gender'] == 'Male']['Survived'].eq('No').sum()) + "** male non-survivors and **" + str(train[train['Gender'] == 'Male']['Survived'].eq('Yes').sum()) + "** male survivors. The gender with more survivors was **" + train[train['Survived'] == 'Yes']['Gender'].mode().iloc[0] + "**. The survival rate was higher for **" + higher_gender + "** passengers.")
        st.subheader("Class Analysis")
        f,ax = plt.subplots(1, 2, figsize=(12, 4))
        (train.assign(Survived= train['Survived'].eq('Yes'))
            [['Class', 'Survived']]
            .groupby(['Class']).mean()
            .plot.bar(ax=ax[0])) #class comparison of survival rate and number of survivors by class
        ax[0].set_title('Survival Rate by Class') 
        ax[0].set_ylabel('Survival Rate')
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=0)
        sns.countplot(x='Class', hue='Survived', data=train, ax=ax[1])
        ax[1].set_ylabel('Quantity') 
        ax[1].set_title('Number of Survivors by Class') 
        st.pyplot(f)
        first_rate = train[train['Class'] == 'First']['Survived'].eq('Yes').mean()
        second_rate = train[train['Class'] == 'Second']['Survived'].eq('Yes').mean()
        third_rate = train[train['Class'] == 'Third']['Survived'].eq('Yes').mean()
        if first_rate > second_rate and first_rate > third_rate:
            highest_class = "First" 
        elif second_rate > first_rate and second_rate > third_rate:
            highest_class = "Second"            
        elif third_rate > first_rate and third_rate > second_rate:
            highest_class ="Third"
        if third_rate < second_rate and third_rate < first_rate:
            lowest_class = "Third" 
        elif second_rate < first_rate and second_rate < third_rate:
            lowest_class = "Second"            
        elif first_rate < third_rate and first_rate < second_rate:
            lowest_class ="First"
        with st.expander("See Insights"):
            st.write("The bar chart shows that the survival rate for first class passengers was **" + str(round(first_rate * 100, 1)) + "%**, while the survival rate for second class passengers was **" + str(round(second_rate * 100, 1)) + "%** and the survival rate for third class passengers was **" + str(round(third_rate * 100, 1)) + "%**. The count plot shows that there were **" + str(train[train['Class'] == 'First']['Survived'].eq('No').sum()) + "** first class non-survivors and **" + str(train[train['Class'] == 'First']['Survived'].eq('Yes').sum()) + "** first class survivors in the dataset. Meanwhile, there were **" + str(train[train['Class'] == 'Second']['Survived'].eq('No').sum()) + "** second class non-survivors and **" + str(train[train['Class'] == 'Second']['Survived'].eq('Yes').sum()) + "** second class survivors. Finally, there were **" + str(train[train['Class'] == 'Third']['Survived'].eq('No').sum()) + "** third class non-survivors and **" + str(train[train['Class'] == 'Third']['Survived'].eq('Yes').sum()) + "** third class survivors. The highest survival rate was for **" + highest_class + "** class passengers, while the lowest survival rate was for **" + lowest_class + "** class passengers. The class with most survivors was **" + train[train['Survived'] == 'Yes']['Class'].mode().iloc[0] + "**.")
    with col2:
        st.subheader("Age Analysis")
        f, ax = plt.subplots(1, 1, figsize=(12, 8))
        #age comparison of number of survivors by age
        sns.histplot(x='Age (Years)', hue='Survived', data=train, ax=ax, bins=20, kde=True)
        ax.set_ylabel('Quantity') 
        ax.set_title('Number of Survivors by Age')
        st.pyplot(f)
        skewness = train[train['Survived'] == 'Yes']['Age (Years)'].skew()
        if skewness > 0:
            direction = "positively (right) skewed"
        elif skewness < 0:
            direction = "negatively (left) skewed"
        else:
            direction = "approximately symmetric"
        if skewness >0:
            age="younger age groups had a higher chance"
        elif skewness<0:
            age="older age groups had a higher chance"
        else: 
            age="age had limited or no effect on survival"
        with st.expander("See Insights"):
            st.write("The histogram shows that the average survival age was **" + str(round(train[train['Survived'] == 'Yes']['Age (Years)'].mean(), 1)) + "** years, while the variation in survival based on the age was **" + str(round(train[train['Survived'] == 'Yes']['Age (Years)'].std(), 1)) + "**. The median age of survivors was **" + str(train[train['Survived'] == 'Yes']['Age (Years)'].median()) + "** years, while the most common age among survivors is **" + str(train[train['Survived'] == 'Yes']['Age (Years)'].mode().iloc[0]) + "** years. The ages of survivors ranged from **" + str(round(train[train['Survived'] == 'Yes']['Age (Years)'].min(), 0)) + "** to **" + str(round(train[train['Survived'] == 'Yes']['Age (Years)'].max(), 0)) + "** years. The age distribution of survivors is **" + direction + "** (skewness = **" + str(round(skewness, 2)) + "**), suggesting that **" +age+ "** of survival.")
        st.subheader("Fare Analysis")
        f,ax = plt.subplots(1, 1, figsize=(12, 8))
        #fare comparison of number of survivors by fare
        sns.histplot(x='Fare (GBP)', hue='Survived', data=train, ax=ax, bins=40, kde=True)
        ax.set_title('Number of Survivors by Fare') 
        ax.set_ylabel('Quantity')
        st.pyplot(f)
        skewness1 = train[train['Survived'] == 'Yes']['Fare (GBP)'].skew()
        if skewness1 > 0:
            direction1 = "positively (right) skewed"
        elif skewness1 < 0:
            direction1 = "negatively (left) skewed"
        else:
            direction1 = "approximately symmetric"
        if skewness1 >0:
            fare="people with higher fareband had a higher chance"
        elif skewness1<0:
            fare="people with lower fareband had a higher chance"
        else: 
            fare="fare had limited or no effect on survival"
        with st.expander("See Insights"):
            st.write("The histogram shows that the average survival fare was **" + str(round(train[train['Survived'] == 'Yes']['Fare (GBP)'].mean(), 1)) + "** GBP, while the variation in survival based on the fare was **" + str(round(train[train['Survived'] == 'Yes']['Fare (GBP)'].std(), 1)) + "**. The median fare of survivors was **" + str(train[train['Survived'] == 'Yes']['Fare (GBP)'].median()) + "** GBP, while the most common fare among survivors is **" + str(train[train['Survived'] == 'Yes']['Fare (GBP)'].mode().iloc[0]) + "** GBP. The fare of survivors ranged from **" + str(round(train[train['Survived'] == 'Yes']['Fare (GBP)'].min(), 0)) + "** to **" + str(round(train[train['Survived'] == 'Yes']['Fare (GBP)'].max(), 0)) + "** GBP. The fare distribution of survivors is **" + direction1 + "** (skewness = **" + str(round(skewness1, 2)) + "**), suggesting that **" +fare+ "** of survival.")
    with col3:
        st.subheader("Embarked Analysis")
        f, ax = plt.subplots(1, 2, figsize=(16, 8))
        (train.assign(Survived= train['Survived'].eq('Yes'))
            [['Embarked', 'Survived']]
            .groupby(['Embarked']).mean()
            .plot.bar(ax=ax[0])) #embarked comparison of survival rate and number of survivors by embarked location
        ax[0].set_title('Survival Rate by Embarked Location')
        ax[0].set_ylabel('Survival Rate')
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=0)
        sns.countplot(x='Embarked', hue='Survived', data=train, ax=ax[1])
        ax[1].set_title('Number of Survivors by Embarked Location') 
        ax[1].set_ylabel('Quantity')
        st.pyplot(f)
        c_rate = train[train['Embarked'] == 'Cherbourg']['Survived'].eq('Yes').mean()
        q_rate = train[train['Embarked'] == 'Queenstown']['Survived'].eq('Yes').mean()
        s_rate = train[train['Embarked'] == 'Southampton']['Survived'].eq('Yes').mean()
        if s_rate > c_rate and s_rate > q_rate:
            highest_embarked = "Southampton"
        elif c_rate > s_rate and c_rate > q_rate:
            highest_embarked = "Cherbourg"
        elif q_rate > s_rate and q_rate > c_rate:
            highest_embarked = "Queenstown"
        if s_rate < c_rate and s_rate < q_rate:
            lowest_embarked = "Southampton"
        elif c_rate < s_rate and c_rate < q_rate:
            lowest_embarked = "Cherbourg"
        elif q_rate < c_rate and q_rate < s_rate:
            lowest_embarked = "Queenstown"
        with st.expander("See Insights"):
            st.write("The bar chart shows that the survival rate for passengers who embarked from **Southampton** was **" + str(round(train[train['Embarked'] == 'Southampton']['Survived'].eq('Yes').mean() * 100, 1)) + "%**, while the survival rate for passengers who embarked from **Cherbourg** was **" + str(round(train[train['Embarked'] == 'Cherbourg']['Survived'].eq('Yes').mean() * 100, 1)) + "%** and the survival rate for passengers who embarked from **Queenstown** was **" + str(round(train[train['Embarked'] == 'Queenstown']['Survived'].eq('Yes').mean() * 100, 1)) + "%**. The count plot shows that there were **" + str(train[train['Embarked'] == 'Southampton']['Survived'].eq('No').sum()) + "** non-survivors and **" + str(train[train['Embarked'] == 'Southampton']['Survived'].eq('Yes').sum()) + "** survivors who embarked from Southampton. Meanwhile, there were **" + str(train[train['Embarked'] == 'Cherbourg']['Survived'].eq('No').sum()) + "** non-survivors and **" + str(train[train['Embarked'] == 'Cherbourg']['Survived'].eq('Yes').sum()) + "** survivors who embarked from Cherbourg. Finally, there were **" + str(train[train['Embarked'] == 'Queenstown']['Survived'].eq('No').sum()) + "** non-survivors and **" + str(train[train['Embarked'] == 'Queenstown']['Survived'].eq('Yes').sum()) + "** survivors who embarked from Queenstown. The highest survival rate was for **" + highest_embarked + "**  passengers, while the lowest survival rate was for **" + lowest_embarked + "**  passengers. The location with most survivors embarked was **" + train[train['Survived'] == 'Yes']['Embarked'].mode().iloc[0] + "**.")
    with col4:
        st.subheader("Sibling/Spouse On Board Analysis")
        f, ax = plt.subplots(1, 2, figsize=(16, 8))
        (train.assign(Survived= train['Survived'].eq('Yes'))
            [['Siblings_Spouse_On_Board', 'Survived']]
            .groupby(['Siblings_Spouse_On_Board']).mean()
            .plot.bar(ax=ax[0])) #siblings spouse on board comparison of survival rate and number of survivors with siblings or spouse on board
        ax[0].set_title('Survival Rate with Siblings/Spouses On Board')
        ax[0].set_ylabel('Survival Rate')
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=0)
        sns.countplot(x='Siblings_Spouse_On_Board', hue='Survived', data=train, ax=ax[1])
        ax[1].set_title('Number of Survivors with Siblings/Spouses On Board') 
        ax[1].set_ylabel('Quantity')
        st.pyplot(f)
        sib0 = train[train['Siblings_Spouse_On_Board'] == 0]['Survived'].eq('Yes').mean()
        sib1 = train[train['Siblings_Spouse_On_Board'] == 1]['Survived'].eq('Yes').mean()
        sib2 = train[train['Siblings_Spouse_On_Board'] == 2]['Survived'].eq('Yes').mean()
        sib4 = train[train['Siblings_Spouse_On_Board'].isin([3,4])]['Survived'].eq('Yes').mean()
        sibinf = train[train['Siblings_Spouse_On_Board'] >= 5]['Survived'].eq('Yes').mean()
        rates = {0: sib0, 1: sib1, 2: sib2, 3: sib4, 5: sibinf}
        highest_sib = max(rates, key=rates.get)
        lowest_sib = min(rates, key=rates.get)
        if highest_sib == 5:
            highest_sib = "5 or more"
        if lowest_sib == 5:
            lowest_sib = "5 or more"
        if highest_sib == 0:
            highest_sib = "no"
        if lowest_sib == 0:
            lowest_sib = "no"
        if sib0 < sib1 and sib0 <sib2:
            reason="Passengers with no siblings or spouse had a lower chance of survival than the ones with one or two siblings or spouse."
        elif sib0 > sib1 and sib0 > sib2:
            reason="Passengers with no siblings or spouse had a higher chance of survival than the ones with one or two siblings or spouse."  
        else:
            reason="Having one or two siblings/spouse had no effect on the passenger survival."
        if sib0 < sib4 and sib0 <sibinf:
            reason1="Passengers with no siblings or spouse had a lower chance of survival than the ones with more than two siblings or spouse."
        elif sib0 > sib4 and sib0 > sibinf:
            reason1="Passengers with no siblings or spouse had a higher chance of survival than the ones with more than two siblings or spouse."  
        else:
            reason1="Having more than two siblings/spouse had no effect on the passenger survival."
        surv0 = train[(train['Siblings_Spouse_On_Board'] == 0) & (train['Survived'] == 'Yes')].shape[0]
        surv1 = train[(train['Siblings_Spouse_On_Board'] == 1) & (train['Survived'] == 'Yes')].shape[0]
        surv2 = train[(train['Siblings_Spouse_On_Board'] == 2) & (train['Survived'] == 'Yes')].shape[0]
        surv3 = train[(train['Siblings_Spouse_On_Board'] == 3) & (train['Survived'] == 'Yes')].shape[0]
        surv4 = train[(train['Siblings_Spouse_On_Board'] == 4) & (train['Survived'] == 'Yes')].shape[0]
        surv5 = train[(train['Siblings_Spouse_On_Board'] >= 5) & (train['Survived'] == 'Yes')].shape[0]
        nsurv0 = train[(train['Siblings_Spouse_On_Board'] == 0) & (train['Survived'] == 'No')].shape[0]
        nsurv1 = train[(train['Siblings_Spouse_On_Board'] == 1) & (train['Survived'] == 'No')].shape[0]
        nsurv2 = train[(train['Siblings_Spouse_On_Board'] == 2) & (train['Survived'] == 'No')].shape[0]
        nsurv3 = train[(train['Siblings_Spouse_On_Board'] == 3) & (train['Survived'] == 'No')].shape[0]
        nsurv4 = train[(train['Siblings_Spouse_On_Board'] == 4) & (train['Survived'] == 'No')].shape[0]
        nsurv5 = train[(train['Siblings_Spouse_On_Board'] >= 5) & (train['Survived'] == 'No')].shape[0]
        with st.expander("See Insights"):
            st.write("The bar chart shows that passengers travelling with **no siblings/spouses** had a survival rate of **" + str(round(sib0 * 100,1)) +
                "%**, while the passengers travelling with **1 sibling/spouse** had a survival rate of **" + str(round(sib1 * 100,1)) +
                "%**. Passengers travelling with **2 siblings/spouses** had a survival rate of **" + str(round(sib2 * 100,1)) +
                "%**. Meanwhile, passengers travelling with **3-4 siblings/spouses** had a survival rate of **" + str(round(sib4 * 100,1)) +
                "%**. Lastly, passengers travelling with **more than 5 siblings/spouses** had a survival rate of **" + str(round(sibinf * 100,1)) +
                "%**. The highest survival rate was for passengers travelling with **" +
                str(highest_sib) + "** sibling(s)/spouse(s), while the lowest survival rate was for passengers travelling with **" +
                str(lowest_sib) + "** sibling(s)/spouse(s). " + str(reason) + " " + str(reason1) +
                " There were **" + str(surv0) + "** survivors and **" + str(nsurv0) + "** non-survivors with **no siblings/spouse**, while there were **" +
                str(surv1) + "** survivors and **" + str(nsurv1) + "** non-survivors with **1 sibling/spouse**, **" + str(surv2) +
                "** survivors and **" + str(nsurv2) + "** non-survivors with **2 siblings/spouses**, **" + str(surv3) +
                "** survivors and **" + str(nsurv3) + "** non-survivors with **3 siblings/spouses**, **" + str(surv4) +
                "** survivors and **" + str(nsurv4) + "** non-survivors with **4 siblings/spouses**, and **" + str(surv5) +
                "** survivors and **" + str(nsurv5) + "** non-survivors with **5 or more siblings/spouses**.")
        st.subheader("Parents/Children On Board Analysis")
        f, ax = plt.subplots(1, 2, figsize=(16, 8))
        (train.assign(Survived= train['Survived'].eq('Yes'))
            [['Parents_Children_On_Board', 'Survived']]
            .groupby(['Parents_Children_On_Board']).mean()
            .plot.bar(ax=ax[0])) #parents children on board comparison of survival rate and number of survivors with parents or children on board
        ax[0].set_title('Survival Rate by Parents/Children On Board')
        ax[0].set_ylabel('Survival Rate')
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=0)
        sns.countplot(x='Parents_Children_On_Board', hue='Survived', data=train, ax=ax[1])
        ax[1].set_title('Number of Survivors with Parents/Children On Board') 
        ax[1].set_ylabel('Quantity')
        st.pyplot(f)
        par0 = train[train['Siblings_Spouse_On_Board'] == 0]['Survived'].eq('Yes').mean()
        par1 = train[train['Siblings_Spouse_On_Board'] == 1]['Survived'].eq('Yes').mean()
        par2 = train[train['Siblings_Spouse_On_Board'] == 2]['Survived'].eq('Yes').mean()
        par3 = train[train['Siblings_Spouse_On_Board'] == 3]['Survived'].eq('Yes').mean()
        par4 = train[train['Siblings_Spouse_On_Board'] == 4]['Survived'].eq('Yes').mean()
        par5 = train[train['Siblings_Spouse_On_Board'] >= 5]['Survived'].eq('Yes').mean()
        rates = {0: par0, 1: par1, 2: par2, 3: par3, 4: par4, 5: par5}
        highest_par = max(rates, key=rates.get)
        lowest_par = min(rates, key=rates.get)
        if highest_par == 5:
            highest_par = "5 or more"
        if lowest_par == 5:
            lowest_par = "5 or more"
        if highest_par == 0:
            highest_par = "no"
        if lowest_par == 0:
            lowest_par = "no"
        if par0 < par1 and par0 <par2 and par0 < par3:
            reason="Passengers with no parents or children had a lower chance of survival than the ones with less than four parents or children."
        elif par0 > par1 and par0 > par2 and par0 > par3:
            reason="Passengers with no parents or children had a higher chance of survival than the ones with less than four parents or children."  
        else:
            reason="Having less than four parents/children had no effect on the passenger survival."
        if par0 <par4 and par0 <par5:
            reason1="Passengers with no parents or children had a lower chance of survival than the ones with more than three three parents or children."
        elif par0 > par4 and par0 > par5:
            reason1="Passengers with no parents or children had a higher chance of survival than the ones with more than three parents or children."  
        else:
            reason1="Having more than four parents/children had no effect on the passenger survival."
        surv0 = train[(train['Parents_Children_On_Board'] == 0) & (train['Survived'] == 'Yes')].shape[0]
        surv1 = train[(train['Parents_Children_On_Board'] == 1) & (train['Survived'] == 'Yes')].shape[0]
        surv2 = train[(train['Parents_Children_On_Board'] == 2) & (train['Survived'] == 'Yes')].shape[0]
        surv3 = train[(train['Parents_Children_On_Board'] == 3) & (train['Survived'] == 'Yes')].shape[0]
        surv4 = train[(train['Parents_Children_On_Board'] == 4) & (train['Survived'] == 'Yes')].shape[0]
        surv5 = train[(train['Parents_Children_On_Board'] >= 5) & (train['Survived'] == 'Yes')].shape[0]
        nsurv0 = train[(train['Parents_Children_On_Board'] == 0) & (train['Survived'] == 'No')].shape[0]
        nsurv1 = train[(train['Parents_Children_On_Board'] == 1) & (train['Survived'] == 'No')].shape[0]
        nsurv2 = train[(train['Parents_Children_On_Board'] == 2) & (train['Survived'] == 'No')].shape[0]
        nsurv3 = train[(train['Parents_Children_On_Board'] == 3) & (train['Survived'] == 'No')].shape[0]
        nsurv4 = train[(train['Parents_Children_On_Board'] == 4) & (train['Survived'] == 'No')].shape[0]
        nsurv5 = train[(train['Parents_Children_On_Board'] >= 5) & (train['Survived'] == 'No')].shape[0]
        with st.expander("See Insights"):
            st.write("The bar chart shows that passengers travelling with **no parents/children** had a survival rate of **" + str(round(par0 * 100,1)) +
            "%**, while the passengers travelling with **1 parent/child** had a survival rate of **" + str(round(par1 * 100,1)) +
            "%**. Passengers travelling with **2 parents/children** had a survival rate of **" + str(round(par2 * 100,1)) +
            "%**. Meanwhile, passengers travelling with **3 parents/children** had a survival rate of **" + str(round(par3 * 100,1)) +
            "%**. Passengers travelling with **4 parents/children** had a survival rate of **" + str(round(par4 * 100,1)) +
            "%**. Lastly, passengers travelling with **5 or more parents/children** had a survival rate of **" + str(round(par5 * 100,1)) +
            "%**. The highest survival rate was for passengers travelling with **" +
            str(highest_par) + "** parent(s)/child(ren), while the lowest survival rate was for passengers travelling with **" +
            str(lowest_par) + "** parent(s)/child(ren). " + str(reason) + " " + str(reason1) +
            " There were **" + str(surv0) + "** survivors and **" + str(nsurv0) + "** non-survivors with **no parents/children**, while there were **" +
            str(surv1) + "** survivors and **" + str(nsurv1) + "** non-survivors with **1 parent/child**, **" + str(surv2) +
            "** survivors and **" + str(nsurv2) + "** non-survivors with **2 parents/children**, **" + str(surv3) +
            "** survivors and **" + str(nsurv3) + "** non-survivors with **3 parents/children**, **" + str(surv4) +
            "** survivors and **" + str(nsurv4) + "** non-survivors with **4 parents/children**, and **" + str(surv5) +
            "** survivors and **" + str(nsurv5) + "** non-survivors with **5 or more parents/children**.")
        
train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)  #feature engineering, optimising data for model training, for example remove unwanted columns like the Cabin.
train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1) #remove Ticket column
train = train.fillna({"Embarked": "Southampton"}) # replacing the missing values in the Embarked feature with Southampton
train["Age (Years)"] = train["Age (Years)"].fillna(-0.5)
test["Age (Years)"] = test["Age (Years)"].fillna(-0.5)
bins = [-1, 0, 5, 12, 20, 24, 40, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager',
          'Student', 'Young Adult', 'Older Adult', 'Senior Citizen']
train['AgeGroup'] = pd.cut(train["Age (Years)"], bins, labels=labels)
test['AgeGroup'] = pd.cut(test["Age (Years)"], bins, labels=labels) #sort the ages into categories
combine = [train, test] #combined group

Survived = {"Yes": 1, "No": 0} #yes no mapping
for dataset in combine:
    if "Survived" in dataset.columns:
        dataset["Survived"] = (
            dataset["Survived"]
            .astype(str)
            .str.strip()
            .map(Survived)
        )

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(train['Title'], train['Gender']) #extract title 
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
                                                 'Don', 'Dr', 'Major',
                                                 'Rev', 'Jonkheer', 'Dona'],
                                                'Rare') #replace uncmmon titles with rare
    dataset['Title'] = dataset['Title'].replace(
        ['Countess', 'Lady', 'Sir'], 'Royal') #replace different royal titles with royal
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
with tab1:
    with col3:
        st.subheader("Title Analysis")
        f,ax = plt.subplots(1, 1, figsize=(12, 8))
        #title comparison of number of survivors by title
        sns.countplot(x='Title', hue='Survived', data=train.replace({'Survived': {0: 'No', 1: 'Yes'}}), ax=ax)
        ax.set_title('Number of Survivors by Title')
        ax.set_ylabel('Quantity')
        st.pyplot(f)
        mr = train[train['Title'] == 'Mr']['Survived'].eq(1).mean()
        miss = train[train['Title'] == 'Miss']['Survived'].eq(1).mean()
        mrs = train[train['Title'] == 'Mrs']['Survived'].eq(1).mean()
        master = train[train['Title'] == 'Master']['Survived'].eq(1).mean()
        royal = train[train['Title'] == 'Royal']['Survived'].eq(1).mean()
        rare = train[train['Title'] == 'Rare']['Survived'].eq(1).mean()
        if mr > miss and mr > mrs and mr > master and mr > royal and mr > rare:
            highest_title = "Mr"
        elif miss > mr and miss > mrs and miss > master and miss > royal and miss > rare:
            highest_title = "Miss"
        elif mrs > mr and mrs > miss and mrs > master and mrs > royal and mrs > rare:
            highest_title = "Mrs"
        elif master > mr and master > miss and master > mrs and master > royal and master > rare:
            highest_title = "Master"
        elif royal > mr and royal > miss and royal > mrs and royal > master and royal > rare:
            highest_title = "Royal"
        else:
            highest_title = "Rare"
        if mr < miss and mr < mrs and mr < master and mr < royal and mr < rare:
            lowest_title = "Mr"
        elif miss < mr and miss < mrs and miss < master and miss < royal and miss < rare:
            lowest_title = "Miss"
        elif mrs < mr and mrs < miss and mrs < master and mrs < royal and mrs < rare:
            lowest_title = "Mrs"
        elif master < mr and master < miss and master < mrs and master < royal and master < rare:
            lowest_title = "Master"
        elif royal < mr and royal < miss and royal < mrs and royal < master and royal < rare:
            lowest_title = "Royal"
        else:
            lowest_title = "Rare"
        with st.expander("See Insights"):
            st.write("The count plot shows that the survival rate for passengers with the title **" + train[train['Title'] == 'Mr']['Title'].head(1).values[0] + "** was **" + str(round(train[train['Title'] == 'Mr']['Survived'].eq(1).mean() * 100, 1)) + "%**, while the survival rate for passengers with the title **Miss** was **" + str(round(train[train['Title'] == 'Miss']['Survived'].eq(1).mean() * 100, 1)) + "%**. The survival rate for passengers with the title **Mrs** was **" + str(round(train[train['Title'] == 'Mrs']['Survived'].eq(1).mean() * 100, 1)) + "%**, while the survival rate for passengers with the title **Master** was **" + str(round(train[train['Title'] == 'Master']['Survived'].eq(1).mean() * 100, 1)) + "%**. The survival rate for passengers with the title **Royal** was **" + str(round(train[train['Title'] == 'Royal']['Survived'].eq(1).mean() * 100, 1)) + "%**, while the survival rate for passengers with the title **Rare** was **" + str(round(train[train['Title'] == 'Rare']['Survived'].eq(1).mean() * 100, 1)) + "%**. The highest survival rate was for the passengers with the title **" + highest_title + "**, while the lowest survival rate was for the passengers with the title **" + lowest_title + "**. The title with most survivors was **" + train[train['Survived'] == 1]['Title'].mode().iloc[0] + "**.")

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3,
                 "Master": 4, "Royal": 5, "Rare": 6} #map each into numerical values
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

mr_age = train[train["Title"] == 1]["AgeGroup"].mode()  # Young Adult
miss_age = train[train["Title"] == 2]["AgeGroup"].mode()  # Student
mrs_age = train[train["Title"] == 3]["AgeGroup"].mode()  # Older Adult
master_age = train[train["Title"] == 4]["AgeGroup"].mode()  # Baby
royal_age = train[train["Title"] == 5]["AgeGroup"].mode()  # Older Adult
rare_age = train[train["Title"] == 6]["AgeGroup"].mode()  # Older Adult

age_title_mapping = {1: "Young Adult", 2: "Student",
                     3: "Older Adult", 4: "Baby", 5: "Older Adult", 6: "Older Adult"}

for x in range(len(train["AgeGroup"])):
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]

for x in range(len(test["AgeGroup"])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3,
               'Student': 4, 'Young Adult': 5, 'Older Adult': 6, 
               'Senior Citizen': 7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

train.head()
train = train.drop(['Age (Years)'], axis=1) #age map
test = test.drop(['Age (Years)'], axis=1)
train = train.drop(['Name'], axis=1) #remove Name
test = test.drop(['Name'], axis=1)
gender_mapping = {"Male": 0, "Female": 1}
train['Gender'] = train['Gender'].map(gender_mapping)
test['Gender'] = test['Gender'].map(gender_mapping) #map gender

embarked_mapping = {"Southampton": 1, "Cherbourg": 2, "Queenstown": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)  #map embarked for assingning numericals

for x in range(len(test["Fare (GBP)"])):
    if pd.isnull(test["Fare (GBP)"][x]):
        pclass = test["Class"][x]  #fill in missing fare, third class if null
        test["Fare (GBP)"][x] = round(
            train[train["Class"] == pclass]["Fare (GBP)"].mean(), 4)


train['FareBand'] = pd.qcut(train['Fare (GBP)'], 4, 
                            labels=[1, 2, 3, 4]) #fare mapping
test['FareBand'] = pd.qcut(test['Fare (GBP)'], 4, 
                           labels=[1, 2, 3, 4]) #fareband 1-4

train = train.drop(['Fare (GBP)'], axis=1)
test = test.drop(['Fare (GBP)'], axis=1)
#map class
class_mapping = {"Third": 3, "Second": 2,"First":1}
train['Class'] = train['Class'].map(class_mapping)
test['Class'] = test['Class'].map(class_mapping) 

from sklearn.model_selection import train_test_split
predictors = train.drop(['Survived', 'Passenger_ID'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(
    predictors, target, test_size=0.2, random_state=0)  #train model 80% train 20% test 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score  #model selection 
#random forest used
randomforest = RandomForestClassifier(n_estimators=100, random_state=42) #it is a supervised machine learning algorithm where multiple decision trees are used
#accuracy remaines same
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.header("Model Predictions")
        st.write("This section provides the predictions of the Random Forest model on the test data, where the survival outcomes are not documented. The accuracy of the model is also displayed.")
        acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
        st.metric("Random Forest Accuracy", f"{acc_randomforest}%") #performance evaluation for accuracy
        ids = test['Passenger_ID']
        predictions = randomforest.predict(test.drop('Passenger_ID', axis=1))
        output = pd.DataFrame({'Passenger_ID': ids, 'Survived': predictions}) #predict the data 

        st.subheader("Titanic Survival Predictions for Passengers 892-1309")
        #yes and no mapping for predictions
        output['Survived'] = output['Survived'].map({0: 'No', 1: 'Yes'})
        with st.expander("See Predictions"):
            st.dataframe(output) #output of ID and predictions
        output.to_csv('Titanic_Results.csv', index=False) #output csv 
        with open("Titanic_Results.csv", "rb") as file:
            st.download_button(
                label="Download CSV File",
                data=file,
                file_name="Titanic_Results.csv",
                mime="application/csv"
            )
    with col2:
        st.subheader("Predicted Survival Analysis")
        f,ax= plt.subplots(1, 2, figsize=(12, 4))
        sns.countplot(x='Survived', data=output, ax=ax[0])
        ax[0].set_title('Number of Predicted Survivors and Non-Survivors')
        ax[0].set_ylabel('Quantity')
        ax[0].bar_label(ax[0].containers[0])
        output['Survived'].value_counts().plot.pie( 
        explode=[0, 0.1], autopct='%1.1f%%', ax=ax[1], shadow=False) 
        ax[1].set_title('Percentage Survived') 
        st.pyplot(f)
        st.write("The pie chart shows that only **" + str(round(output['Survived'].eq('Yes').mean() * 100, 1)) + "%** of passengers survived. The bar chart shows that there were **" + str(output['Survived'].eq('No').sum()) + "** non-survivors and **" + str(output['Survived'].eq('Yes').sum()) + "** survivors in the dataset.")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Passengers", len(train) + len(output))

    survivors = train['Survived'].eq(1).sum() + output['Survived'].eq('Yes').sum()
    passengers = len(train) + len(output)

    col2.metric("Survival Rate", f"{round(survivors/passengers*100,1)}%")
    col3.metric("Non-Survivors", train['Survived'].eq(0).sum() + output['Survived'].eq('No').sum())
    col4.metric("Survivors", survivors)
with tab2: 
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Passengers", len(train))

    survivors = train['Survived'].eq(1).sum() 
    passengers = len(train) 

    col2.metric("Survival Rate", f"{round(survivors/passengers*100,1)}%")
    col3.metric("Non-Survivors", train['Survived'].eq(0).sum())
    col4.metric("Survivors", survivors)