#!/usr/bin/env python
# coding: utf-8

# # 1. Importing Library and Dataset

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("/home/hunter/Desktop/stige/application_data.csv")
df.head()


# # 2. Basic Check

# In[3]:


pd.DataFrame({"Rows":df.shape[0],"Columns":df.shape[1]},index = [1])


# In[4]:


pd.DataFrame({"Column Name":df.columns})


# In[5]:


df['SK_ID_CURR'].nunique()


# There is a unique id for every request

# # 2.1 Missing Values

# In[6]:


df.info()


# In[7]:


missing = df.isnull().sum()
missing = missing[missing>0]# Taking values where there are missing values
miss_per = round(missing/len(df),3)*100
pd.DataFrame({"Missing Entries":missing,"Missing Percentage":miss_per}).sort_values("Missing Percentage",ascending = False)


# There are 67 columns with missing values and in some columns its percentage is over 60

# # 2.3 Descriptive Statistics

# In[8]:


df.describe()


# # 2.4 Outlier 

# In[9]:


# #Outlier Treatment
# def outlier_detect(df):
#     for i in df.describe().columns:
#         Q1=df.describe().at['25%',i]
#         Q3=df.describe().at['75%',i]
#         IQR=Q3 - Q1
#         LTV=Q1 - 1.5 * IQR
#         UTV=Q3 + 1.5 * IQR
#         x=np.array(df[i])
#         p=[]
#         for j in x:
#             if j < LTV or j>UTV:
#                 p.append(df[i].median())
#             else:
#                 p.append(j)
#         df[i]=p
#     return df


# In[10]:


# dat = outlier_detect(df)


# # 3 Exploratory Data Analysis (EDA)

# # 3.1 Categorical Variables

# Start with categorical variable analysis as they give brief idea about the behaviour of data

# ##  3.1.1 Target

# In[11]:


df['TARGET'].value_counts(dropna = False)


# In[12]:


plt.figure(figsize = (8,5))
sns.countplot(df['TARGET'])
plt.title("Number of client in each difficulty category")


# Here we could see that one category has a lot more data than other category which clearly shows there is a __Data Imbalance__.
# - Dividing the data into 2 dataframe one where there is no payment difficulty and in other there is payment difficulty.

# In[13]:


df0 = df.loc[df["TARGET"]==0,:]
df0.head()


# In[14]:


df1 = df.loc[df["TARGET"]==1,:]
df1.head()


# In[15]:


def overalldata(dat,colname):
    plt.figure(figsize = (8,5))
    sns.countplot(x=colname,data = dat)
    print(dat[colname].value_counts(dropna = False))


# In[16]:


def paymentdifficulty(dat1,colname):
    plt.figure(figsize = (8,5))
    sns.countplot(x=colname,data = dat1)
    print(dat1[colname].value_counts(dropna = False))


# ## 3.1.2 NAME_CONTRACT_TYPE

# In[17]:


overalldata(df,"NAME_CONTRACT_TYPE")


# - Its clearly seen that mostly Cash loan is prominent.
# - Let's see its association with __TARGET__

# - __No payment Difficulty__

# In[18]:


plt.figure(figsize = (8,5))
sns.countplot(x="NAME_CONTRACT_TYPE",data = df0)


# In[19]:


df0['NAME_CONTRACT_TYPE'].value_counts(dropna = False)


# - Total __Cashloan__ given are 278232 out of which 255011 around __91.65%__ falls in no payment difficulty condition.
# - Total __Revolving loans__ given are 29279 out of which 27675 around __94.21%__ falls in no payment difficulty condition.           
# Here we can see clearly in Revolving loan Category payment difficulty is less

# - __Payment Difficulty__

# In[20]:


paymentdifficulty(df1,"NAME_CONTRACT_TYPE")


# - Total __Cashloan__ given are 278232 out of which 23221 around __8.35%__ falls in  payment difficulty condition.
# - Total __Revolving loans__ given are 29279 out of which 1604 around __5.79%__ falls in  payment difficulty condition.           
# Here we can Confirm fact in Revolving loan Category payment difficulty is less

# ## 3.1.3 CODE_GENDER

# In[21]:


overalldata(df,"CODE_GENDER")


# Female population is more than Male and XNA are very less compare to other so there effect acn be ignored.

# - __Payment Difficulty__

# In[22]:


paymentdifficulty(df1,"CODE_GENDER")


# - Total 202448 Female client were there out of which 14170 around __7%__ have payment difficulty.
# - Total 105059 Male client were there out of which 10655 around __10.14%__ have payment difficulty.                           
# 
# Here it is clearly seen that male clients have more payment difficulty than female one.

# ## 3.1.4 FLAG_OWN_CAR

# In[23]:


overalldata(df,"FLAG_OWN_CAR")


# The population of people without car is more than who possessed car.

# - __Payment Difficulty__

# In[24]:


paymentdifficulty(df1,"FLAG_OWN_CAR")


# - There are total 202924 client with no car out of which 17249 around __8.5%__ have payment difficulty.
# - There are total 104587 client with car out of which 7576 around __7.24%__ have payment difficulty.
# 
# Client not having car has more payment difficulty.

# ## 3.1.5 FLAG_OWN_REALTY

# In[25]:


overalldata(df,"FLAG_OWN_REALTY")


# The population of people with flat is more than who does not.

# - __Payment Difficulty__

# In[26]:


paymentdifficulty(df1,"FLAG_OWN_REALTY")


# - There are total 213312 client with flat out of which 16983 around __7.96%__ have payment difficulty.
# - There are total 94199 client withot flat out of which 7842 around __8.32%__ have payment difficulty.
# 
# Client not having flat has more payment difficulty.
# 

# ## 3.1.6 NAME_INCOME_TYPE

# In[27]:


overalldata(df,"NAME_INCOME_TYPE")


# The major chunk of client belongs to 4 categories Working,Commercial associate,Pensioner, State servant

# - __Payment Difficulty__

# In[28]:


paymentdifficulty(df1,"NAME_INCOME_TYPE")


# - There are total 158774 Working client out of which 15224 around __9.59%__ have payment difficulty.
# - There are total 71017 Commercial associate client out of which 5360 around __7.55%__ have payment difficulty.
# - There are total 55362 Pensioner client out of which 2982 around __5.39%__ have payment difficulty.
# - There are total 21703 State Servant client out of which 1249 around __5.75%__ have payment difficulty.
# - There are total 22 Unemployed client out of which 8 around __36.36%__ have payment difficulty.
# - There are total 5 Maternity leave client out of which 2 around __40%__ have payment difficulty.
# 
# Here it is clearly seen Unemployed and Maternity leave client highest percent of payment difficulty whereas Students and Businessmen has no payment difficulty. 
# 
# 
# 

# ## 3.1.7 NAME_EDUCATION_TYPE

# In[29]:


overalldata(df,"NAME_EDUCATION_TYPE")


# The population of people with Secondary / secondary special is more than others.

# In[30]:


paymentdifficulty(df1,"NAME_EDUCATION_TYPE")


# - There are total 218391 Secondary / secondary special client out of which 19524 around __8.94%__ have payment difficulty.
# - There are total 74863 Higher education client out of which 4009 around __5.35%__ have payment difficulty.
# - There are total 10277 Incomplete higher client out of which 872 around __8.48%__ have payment difficulty.
# - There are total 3816 Lower secondary client out of which 417 around __10.93%__ have payment difficulty.
# - There are total 164 Academic degree client out of which 3 around __1.83%__ have payment difficulty.
# 
# 
# Here it is clearly seen Secondary / secondary special client,Incomplete higher client and Lower secondary client has highest percent of payment difficulty whereas Academic degree has very less payment difficulty. 

# ## 3.1.8 NAME_FAMILY_STATUS

# In[31]:


overalldata(df,"NAME_FAMILY_STATUS")


# The population of people who are married is more than all other.

# - __Payment Difficulty__

# In[32]:


paymentdifficulty(df1,"NAME_FAMILY_STATUS")


# - There are total 196432 Married client out of which 14850 around __7.56%__ have payment difficulty.
# - There are total 45444 Single/not married client out of which 4457 around __9.81%__ have payment difficulty.
# - There are total 29775 Civil marriage client out of which 2961 around __9.94%__ have payment difficulty.
# - There are total 19770 Seperated client out of which 1620 around __8.19%__ have payment difficulty.
# - There are total 16088 Widow client out of which 937 around __5.82%__ have payment difficulty.
# 
# 
# Here it is clearly seen Single/not married client,Civil Marriage client and Seperated client has highest percent of payment difficulty.

# ## 3.1.9 NAME_HOUSING_TYPE

# In[33]:


overalldata(df,"NAME_HOUSING_TYPE")


# The population of people with House/apartment is most among all.

# In[34]:


paymentdifficulty(df1,"NAME_HOUSING_TYPE")


# - There are total 272868 House / apartment client out of which 21272 around __7.80%__ have payment difficulty.
# - There are total 14840 With parents client out of which 1736 around __11.70%__ have payment difficulty.
# - There are total 11183 Municipal apartment client out of which 955 around __8.54%__ have payment difficulty.
# - There are total 4881 Rented apartment client out of which 601 around __12.31%__ have payment difficulty.
# - There are total 2617 Office apartment client out of which 172 around __6.57%__ have payment difficulty.
# - There are total 1122 Co-op apartment client out of which 89 around __7.93%__ have payment difficulty.
# 
# 
# Here it is clearly seen With parents client,Rented apartment client has highest percent of payment difficulty.

# ## 3.1.10 FLAG_MOBIL

# In[35]:


overalldata(df,"FLAG_MOBIL")


# ALmost all possessed mobile so there is no special info could be generated as it will have same value all row and could be the candidate for dropping of column

# ## 3.1.11 FLAG_EMP_PHONE

# In[36]:


overalldata(df,"FLAG_EMP_PHONE")


# Generally client do provide workphone.

# - __Payment Difficulty__

# In[37]:


paymentdifficulty(df1,"FLAG_EMP_PHONE")


# - There are total 252125 client which provided Work phone number out of which 21834 around __8.66%__ have payment difficulty.
# - There are total 55386 client which not provided Work phone number out of which 2991 around __5.40%__ have payment difficulty.
# 
# Client providining workphone has more payment difficulty.
# 

# ## 3.1.12 FLAG_CONT_MOBILE

# In[38]:


overalldata(df,"FLAG_CONT_MOBILE")


# Most of the client phones are reachable

# - __Payment Difficulty__

# In[39]:


paymentdifficulty(df1,"FLAG_CONT_MOBILE")


# - There are total 306937 client whose phone was reachable out of which 24780 around __8.07%__ have payment difficulty.
# - There are total 574 client whose phone was not reachable out of which 45 around __7.83%__ have payment difficulty.
# 
# Both categoties proportion are almost same cannot be distinguished properly.

# # 3.1.13 FLAG_EMAIL

# In[40]:


overalldata(df,"FLAG_EMAIL")


# Major proportion of client does not provide mail.

# - __Payment Difficulty__

# In[41]:


paymentdifficulty(df1,"FLAG_EMAIL")


# - There are total 290069 client who not provude mail out of which 23451 around __8.08%__ have payment difficulty.
# - There are total 17442 client who not provude mail out of which 1374 around __7.88%__ have payment difficulty.
# 
# Here not much difference between two categories.
# 

# ## 3.1.14 OCCUPATION_TYPE

# In[42]:


overalldata(df,"OCCUPATION_TYPE")


# Here lot of data around 32% data is missing which is more than all the category so no concrete information could fe found out

# ## 3.1.15 CNT_FAM_MEMBERS

# In[43]:


overalldata(df,"CNT_FAM_MEMBERS")


# Majorly count of family members varies from 0 t0 5

# - __Payment Difficulty__

# In[44]:


paymentdifficulty(df1,"CNT_FAM_MEMBERS")


# - There are total 158357 2 member client out of which 12009 around __7.58%__ have payment difficulty.
# - There are total 67847 1 member client out of which 5675 around __8.36%__ have payment difficulty.
# - There are total 52601 3 member client out of which 4608 around __8.76%__ have payment difficulty.
# - There are total 24697 4 member client out of which 2136 around __8.65%__ have payment difficulty.
# - There are total 3478 5 member client out of which 327 around __9.41%__ have payment difficulty.
# 
# 
# Here it is clearly seen 2 member percent of payment difficulty is least.

# ## 3.1.16 REGION_RATING_CLIENT

# In[45]:


overalldata(df,"REGION_RATING_CLIENT")


# Major portion of client live rating 2 category

# - __Payment Difficulty__

# In[46]:


paymentdifficulty(df1,"REGION_RATING_CLIENT")


# - There are total 226984 client with rating 2 out of which 17907 around __7.89%__ have payment difficulty.
# - There are total 48330 client with rating 3 out of which 5366 around __11.10%__ have payment difficulty.
# - There are total 32197 client with rating 1 out of which 1552 around __4.82%__ have payment difficulty.
# 
# Here clearly most difficulty with rating 3 client and least among rating 1 clients.

# ## 3.1.17 REG_REGION_NOT_LIVE_REGION

# In[47]:


overalldata(df,"REG_REGION_NOT_LIVE_REGION")


# Majority of client's permanent address match contact address 

# - __Payment Difficulty__

# In[48]:


paymentdifficulty(df1,"REG_REGION_NOT_LIVE_REGION")


# - There are total 302854 client's permanent address match contact address out of which 24392 around __8.05%__ have payment difficulty.
# - There are total 4657 client's permanent address does not match contact address out of which 433 around __9.30%__ have payment difficulty.
# 
# AS contact address does not match permanent address payment difficulty increases.

# ## 3.1.18 REG_REGION_NOT_WORK_REGION

# In[49]:


overalldata(df,"REG_REGION_NOT_WORK_REGION")


# Majority of client's permanent address match Work address

# In[50]:


paymentdifficulty(df1,"REG_REGION_NOT_WORK_REGION")


# - There are total 291899 client's permanent address match work address out of which 23437 around __8.03%__ have payment difficulty.
# - There are total 15612 client's permanent address does not work contact address out of which 1388 around __8.89%__ have payment difficulty.
# 
# AS work address does not match permanent address payment difficulty increases.

# ## 3.1.19 LIVE_REGION_NOT_WORK_REGION

# In[51]:


overalldata(df,"LIVE_REGION_NOT_WORK_REGION")


# Majority of client's contact address match Work address

# In[52]:


paymentdifficulty(df1,"LIVE_REGION_NOT_WORK_REGION")


# - There are total 295008 client's contact address match work address out of which 23769 around __8.06%__ have payment difficulty.
# - There are total 12503 client's contact address does not match work contact address out of which 1056 around __8.45%__ have payment difficulty.
# 
# AS contact address does not match work address payment difficulty increases.

# __CITY LEVEL__

# ## 3.1.20 REG_CITY_NOT_LIVE_CITY

# In[53]:


overalldata(df,"REG_CITY_NOT_LIVE_CITY")


# Majority of client's permanent address match contact address

# - __Payment Difficulty__

# In[54]:


paymentdifficulty(df1,"REG_CITY_NOT_LIVE_CITY")


# - There are total 283472 client's permanent address match contact address out of which 21886 around __7.72%__ have payment difficulty.
# - There are total 24039 client's permanent address does not match contact address out of which 2939 around __12.22%__ have payment difficulty.
# 
# AS contact address does not match permanent address payment difficulty increases.

# ## 3.1.21 REG_CITY_NOT_WORK_CITY

# In[55]:


overalldata(df,"REG_CITY_NOT_WORK_CITY")


# Majority of client's permanent address match Work address

# In[56]:


paymentdifficulty(df1,"REG_CITY_NOT_WORK_CITY")


# - There are total 236644 client's permanent address match work address out of which 17305 around __7.31%__ have payment difficulty.
# - There are total 70867 client's permanent address does not work contact address out of which 7520 around __10.61%__ have payment difficulty.
# 
# AS work address does not match permanent address payment difficulty increases.

# ## 3.1.22 LIVE_CITY_NOT_WORK_CITY

# In[57]:


overalldata(df,"LIVE_CITY_NOT_WORK_CITY")


# Majority of client's contact address match Work address

# In[58]:


paymentdifficulty(df1,"LIVE_CITY_NOT_WORK_CITY")


# - There are total 252296 client's contact address match work address out of which 19322 around __7.66%__ have payment difficulty.
# - There are total 55215 client's contact address does not match work contact address out of which 5503 around __9.97%__ have payment difficulty.
# 
# AS contact address does not match work address payment difficulty increases.

# __As it could be seen diffference at city level is much prominent as compare to region level if addresses does not match.__

# ## 3.1.23 FLAG_DOCUMENT_3

# In[59]:


overalldata(df,"FLAG_DOCUMENT_3")


# we can see that clients who have document 3 are more.

# In[60]:


paymentdifficulty(df1,"FLAG_DOCUMENT_3")


# *  There are total  218340  client who has documet 3 out of which 19312 around __11.30__% have payment difficulty.
# *   There are total  89171
#   client who has documet 3 out of which 5513 around __16.174__% have payment difficulty.
# 
# we can see that the clients who  have document 3 are having more problem in payment.
# 

# ## 3.1.24 FLAG_DOCUMENT_4

# In[61]:


overalldata(df,"FLAG_DOCUMENT_4")


# we can see that clients who dont have document 4 are more
# 

# In[62]:


paymentdifficulty(df1,"FLAG_DOCUMENT_4")


# 
# 
# *    There are total  307486  client who dont have documet 4 out of which 24825 around __13.38__% have payment difficulty.
# *   There are no client who have document 4 and facing the payment difficulty
# 
# 

# ## 3.1.25 FLAG_DOCUMENT_5 

# In[63]:


overalldata(df,"FLAG_DOCUMENT_5")


# we can see that most of the people dont have document 5

# In[64]:


paymentdifficulty(df1,"FLAG_DOCUMENT_5")


# 
# 
# *   There are total   302863  client who dont have documet 5 out of which 24453 around __13.38__% have payment difficulty.
# *   There are total   4648  client who  have documet 5 out of which 372 around __12.49__% have payment difficulty.
# as we can see that clients who have document 5 or dont have document  5 having almost same problem in payment.
# 
# 

# ## 3.1.13 FLAG_DOCUMENT_6

# In[65]:


overalldata(df,"FLAG_DOCUMENT_6")


# There are 280433 clients who dont have document 6 and 27078 have document 6 and we can see that most of the clients dont have document 6

# In[66]:


paymentdifficulty(df1,"FLAG_DOCUMENT_6")


# 
# *   There are total   280433  client who dont have documet 6 out of which 23318 around __12.02__% have payment difficulty.
# *   There are total   27078  client who  have documet 6 out of which 1507 around __17.96__% have payment difficulty.
# we can see that clients who have document 6 having more problem in payment.
# 
# 
# 
# 

# ## 3.1.26 FLAG_DOCUMENT_7

# In[67]:


overalldata(df,"FLAG_DOCUMENT_7")


# we can see that most of the clients dont have Document 7

# In[68]:


paymentdifficulty(df1,"FLAG_DOCUMENT_7")


# 
# *   There are total   307452  client who dont have documet 7 out of which 24822 around __12.38__% have payment difficulty.
# *   There are total   59  client who  have documet 7 out of which 3 around __19.66__% have payment difficulty.
# 
# we can see that clients who have document 7 having more problem in payment.
# 
# 
# 
# 

# ## 3.1.27 FLAG_DOCUMENT_8 

# In[69]:


overalldata(df,"FLAG_DOCUMENT_8")


# In[70]:


paymentdifficulty(df1,"FLAG_DOCUMENT_8")


# 
# *   There are total   282487  client who dont have documet 8 out of which 22989 around __12.28__% have payment difficulty.
# *   There are total   25024  client who  have documet 8 out of which 1836 around __13.62__% have payment difficulty.
# 
# 
# 
# 

# ## 3.1.28 FLAG_DOCUMENT_9

# In[71]:


overalldata(df,"FLAG_DOCUMENT_9")


# In[72]:


paymentdifficulty(df1,"FLAG_DOCUMENT_9")


# 
# *   There are total   306313  client who dont have documet 9 out of which 24751 around __12.37__% have payment difficulty.
# *   There are total   1198  client who  have documet 9 out of which 74 around __16.18__% have payment difficulty.
# 
# we can see that clients who have document 9 having more problem in payment.
# 
# 
# 

# ## 3.1.29 FLAG_DOCUMENT_10

# In[73]:


overalldata(df,"FLAG_DOCUMENT_10")


# In[74]:


paymentdifficulty(df1,"FLAG_DOCUMENT_10")


# 
# *   There are total   307504  client who dont have documet 9 out of which 24825 around __12.38__% have payment difficulty.
# *   There are total   7  client who  have documet 9 and no one is facing the payment difficulty
# 
# 
# 
# 

# ## 3.1.30 FLAG_DOCUMENT_11

# In[75]:


overalldata(df,"FLAG_DOCUMENT_11")


# In[76]:


paymentdifficulty(df1,"FLAG_DOCUMENT_11")


# 
# *   There are total   306308  client who dont have documet 9 out of which 24750 around __12.37__% have payment difficulty.
# *   There are total   1203  client who  have documet 9 out of which 75 around __16.04__% have 
# payment difficulty.
# 
# we can see that clients who have document 11 having more problem in payment.
# 
# 
# 
# 

# ## 3.1.30 FLAG_DOCUMENT_12
# 
# 
# 
# 
# 
# 
# 

# In[77]:


overalldata(df,"FLAG_DOCUMENT_12")


# In[78]:


paymentdifficulty(df1,"FLAG_DOCUMENT_12")


# 
# *   There are total   307509  client who dont have documet 9 out of which 24825 around __12.38__% have payment difficulty.
# *   There are total   2  client who  have documet 9 and no one facing
# payment difficulty.
# 
# 
# 
# 

# ## 3.1.32 FLAG_DOCUMENT_13
# 
# 
# 
# 

# In[79]:


overalldata(df,"FLAG_DOCUMENT_13")


# In[80]:


paymentdifficulty(df1,"FLAG_DOCUMENT_13")


# 
# *   There are total   306427  client who dont have documet 13 out of which 24795 around __12.35__% have payment difficulty.
# *   There are total   1084 client who  have documet 13 and out of which 30  around __36.13__% having
# payment difficulty.
# 
# we can see that clients who have document 13 having more problem in payment. and the number is very high who are not able to do payment
# 

# ## 3.1.33 FLAG_DOCUMENT_14
# 

# In[81]:


overalldata(df,"FLAG_DOCUMENT_14")


# In[82]:


paymentdifficulty(df1,"FLAG_DOCUMENT_14")


# 
# *   There are total   306608  client who dont have documet 14 out of which 24795 around __12.36__% have payment difficulty.
# *   There are total   903 client who  have documet 14 and out of which 30  around __30.10__% having
# payment difficulty.
# 
# we can see that clients who have document 14 having more problem in payment. and the number is very high who are not able to do payment

# ## 3.1.34 FLAG_DOCUMENT_15
# 

# In[83]:


overalldata(df,"FLAG_DOCUMENT_15")


# In[84]:


paymentdifficulty(df1,"FLAG_DOCUMENT_15")


# 
# *   There are total   307139  client who dont have documet 15 out of which 24814 around __12.37__% have payment difficulty.
# *   There are total   372 client who  have documet 15 and out of which 11  around __33.81__% having
# payment difficulty.

# ## 3.1.35 FLAG_DOCUMENT_16

# In[85]:


overalldata(df,"FLAG_DOCUMENT_16")


# In[86]:


paymentdifficulty(df1,"FLAG_DOCUMENT_16")


# 
# *   There are total   304458  client who dont have documet 16 out of which 24675 around __12.33__% have payment difficulty.
# *   There are total   3053 client who  have documet 16 and out of which 150  around __20.42__% having
# payment difficulty.

# ## 3.1.36 FLAG_DOCUMENT_17

# In[87]:


overalldata(df,"FLAG_DOCUMENT_17")


# In[88]:


paymentdifficulty(df1,"FLAG_DOCUMENT_17")


# 
# *   There are total   307429  client who dont have documet 17 out of which 24823 around __12.38__% have payment difficulty.
# *   There are total   82 client who  have documet 17 and out of which 2  around __41__% having
# payment difficulty.
# 
#  we can see that almost half of the clients who have document 17 are not able to do payment

# ## 3.1.37 FLAG_DOCUMENT_18

# In[89]:


overalldata(df,"FLAG_DOCUMENT_18")


# In[90]:


paymentdifficulty(df1,"FLAG_DOCUMENT_18")


# 
# *   There are total   305011  client who dont have documet 18 out of which 24683 around __12.35__% have payment difficulty.
# *   There are total   2500 client who  have documet 18 and out of which 142  around __17.60__% having
# payment difficulty.

# ## 3.1.38 FLAG_DOCUMENT_19

# In[91]:


overalldata(df,"FLAG_DOCUMENT_19")


# In[92]:


paymentdifficulty(df1,"FLAG_DOCUMENT_19")


# 
# *   There are total   307328  client who dont have documet 19 out of which 24813 around __12.38__% have payment difficulty.
# *   There are total   183 client who  have documet 19 and out of which 12  around __15.25__% having
# payment difficulty.

# ## 3.1.39 FLAG_DOCUMENT_20

# In[93]:


overalldata(df,"FLAG_DOCUMENT_20")


# In[94]:


paymentdifficulty(df1,"FLAG_DOCUMENT_20")


# 
# *   There are total   307355  client who dont have documet 20 out of which 24812 around __12.38__% have payment difficulty.
# *   There are total   156 client who  have documet 20 and out of which 13  around __12__% having
# payment difficulty.

# ## 3.1.40 FLAG_DOCUMENT_21

# In[95]:


overalldata(df,"FLAG_DOCUMENT_21")


# In[96]:


paymentdifficulty(df1,"FLAG_DOCUMENT_21")


# 
# *   There are total   307408  client who dont have documet 21 out of which 24811 around __12.38__% have payment difficulty.
# *   There are total   103 client who  have documet 21 and out of which 14  around __7.35__% having
# payment difficulty.
# 
# here the number of clients who are not having document 21 are not able to do payment are more.

# ## 3.1.41 AMT_REQ_CREDIT_BUREAU_HOUR

# In[97]:


overalldata(df,"AMT_REQ_CREDIT_BUREAU_HOUR")


# There are 264366 clients who have 0 queries and these are the maximum among all
# 
# 
# There are lots of NaN values present in this attribute

# In[98]:


paymentdifficulty(df1,"AMT_REQ_CREDIT_BUREAU_HOUR")


# 
# *  there are 264366 clients who having zero query before one hour the applicationout of which 20402 having defficulties in payment that is arround __12.95__%.
# 
# *  there are 1560 clients who having one query before one hour the applicationout of which 125 having defficulties in payment that is arround __12.48__%.
# *  there are 56 clients who having two query before one hour the applicationout of which 6 having defficulties in payment that is arround __9.33__%.  
# 
# 

# # Analyse Previous application Data

# In[99]:


data = pd.read_csv("/home/hunter/Desktop/stige/previous_application.csv")
data.head()


# # Basic Check

# In[100]:


pd.DataFrame({"Rows":data.shape[0],"Columns":data.shape[1]},index = [1])


# In[101]:


pd.DataFrame({"Column Name":data.columns})


# # Missing values

# In[102]:


data.info()


# In[103]:


missing = data.isnull().sum()
missing = missing[missing>0]# Taking values where there are missing values
miss_per = round(missing/len(data),3)*100
pd.DataFrame({"Missing Entries":missing,"Missing Percentage":miss_per}).sort_values("Missing Percentage",ascending = False)


# Lot of missing data is here some columns have missing percentage over 90 that need to be dropped

# # Descriptive Statistics

# In[104]:


data.describe()


# # Exploratory Data Analysis 

# # Categorical Variables

# ## NAME_CONTRACT_STATUS

# In[105]:


overalldata(data,"NAME_CONTRACT_STATUS")


# Mostly application approved and Canceled, Refused offer are similar

# __Let's divide dataset into 4 categories so to analyse each effect.__

# In[106]:


dataA = data.loc[data["NAME_CONTRACT_STATUS"]=="Approved"]
dataA.head()


# In[107]:


dataC = data.loc[data["NAME_CONTRACT_STATUS"]=="Canceled"]
dataC.head()


# In[108]:


dataR = data.loc[data["NAME_CONTRACT_STATUS"]=="Refused"]
dataR.head()


# In[109]:


dataU = data.loc[data["NAME_CONTRACT_STATUS"]=="Unused offer"]
dataU.head()


# ## NAME_CONTRACT_TYPE

# In[110]:


overalldata(data,"NAME_CONTRACT_TYPE")


# The occurence of consumer loans and cash loan are similar whereas Revolving loans are less compare to others.

# - __APPROVED__

# In[111]:


overalldata(dataA,"NAME_CONTRACT_TYPE")


# - There are total 747553 client with Cash loan out of which 312540 around __41.81%__ are approved.
# - There are total 729151 client with Consumer loan out of which 626470 around __85.92%__ are approved.
# - There are total 193164 client with Revolving loan out of which 97771 around __50.61%__ are approved.
# 
# For consumer loan Approval is most.

# - __CANCELED__

# In[112]:


overalldata(dataC,"NAME_CONTRACT_TYPE")


# - There are total 747553 client with Cash loan out of which 268591 around __35.93%__ are Canceled.
# - There are total 729151 client with Consumer loan out of which 45854 around __6.29%__ are Canceled.
# - There are total 193164 client with Revolving loan out of which 1559 around __0.81%__ are Canceled.
# 
# For Cash loan Cancellation is most.

# - __REFUSED__

# In[113]:


overalldata(dataR,"NAME_CONTRACT_TYPE")


# - There are total 747553 client with Cash loan out of which 165928 around __22.20%__ are Refused.
# - There are total 747553 client with Consumer loan out of which 75185 around __10.06%__ are Refused.
# - There are total 747553 client with Revolving loan out of which 49534 around __6.63%__ are Refused.
# 
# For Cash loan Refuse is most.

# - __UNUSED OFFER__

# In[114]:


overalldata(dataU,"NAME_CONTRACT_TYPE")


# - There are total 747553 client with Cash loan out of which 494 around __0.067%__ are Unused Offer.
# - There are total 747553 client with Consumer loan out of which 25937 around __3.47%__ are Unused Offer.
# - There are negligible unused offers for revolving loan.
# 
# For Consumer loan Unused offer is most.

# ## WEEKDAY_APPR_PROCESS_START

# In[115]:


overalldata(data,"WEEKDAY_APPR_PROCESS_START")


# Generally on the week day client apply for previous application as compare to weekend.

# ## FLAG_LAST_APPL_PER_CONTRACT

# In[116]:


overalldata(data,"FLAG_LAST_APPL_PER_CONTRACT")


# Mostly there is no mistake for more than application.

# ## NFLAG_LAST_APPL_IN_DAY

# In[117]:


overalldata(data,"NFLAG_LAST_APPL_IN_DAY")


# Mostly it is one application for client per day.

# # NAME_PORTFOLIO

# In[118]:


overalldata(data,"NAME_PORTFOLIO")


# # approved

# In[119]:


overalldata(dataA,"NAME_PORTFOLIO")


# - There are total 691011 pos out of which 626207 approved
# - There are total 461563 cash out of which 312536 approved
# - There are total 372230 Xna out of which 4 approved
# - There are total 144985 cards out of which 97771 approved
# - There are total 425 cars out of which 263 approved

# # Refused

# In[120]:


overalldata(dataR,"NAME_PORTFOLIO")


# - There are total 691011 pos out of which 63720 refused
# - There are total 461563 cash out of which 139204 refused
# - There are total 372230 Xna out of which 40897 refused
# - There are total 144985 cards out of which 46739 refused
# - There are total 425 cars out of which 118 refused

# # Unused Name Portfolio

# In[121]:


overalldata(dataU,"NAME_PORTFOLIO")


# - There are total 691011 pos out of which 910 refused
# - There are total 461563 cash out of which 139204 refused
# - There are total 372230 Xna out of which 0 refused
# - There are total 144985 cards out of which 2 refused
# - There are total 425 cars out of which 0 refused

# # Canceled name portforlio

# In[122]:


overalldata(dataC,"NAME_PORTFOLIO")


# - There are total 691011 pos out of which 174 canceled
# - There are total 461563 cash out of which 9823 canceled
# - There are total 372230 Xna out of which 305805 canceled
# - There are total 144985 cards out of which 473 canceled
# - There are total 425 cars out of which 44 canceled

# # NAME_PRODUCT_TYPE

# In[123]:


overalldata(data,"NAME_PRODUCT_TYPE")


# - there are lots of XNA  and the walk ins are very less

# - approved

# In[124]:


overalldata(dataA,"NAME_PRODUCT_TYPE")


# - There are total 1063666 XNA out of which 626474 approved
# - There are total 456287 x-sell out of which 337649 approved
# - There are total 150261 walk-in out of which 72658 approved
# 

# - Refused

# In[125]:


overalldata(dataR,"NAME_PRODUCT_TYPE")


# - There are total 1063666 XNA out of which 104735 refused
# - There are total 456287 x-sell out of which 110201 refused
# - There are total 150261 walk-in out of which 75742 refused
# 

# - canceled

# In[126]:


overalldata(dataC,"NAME_PRODUCT_TYPE")


# - There are total 1063666 XNA out of which 306023 cancled
# - There are total 456287 x-sell out of which 8435 canceled
# - There are total 150261 walk-in out of which 1861 canceld
# 

# - unused

# In[127]:


overalldata(dataU,"NAME_PRODUCT_TYPE")


# - There are total 1063666 XNA out of which 26434 cancled
# - There are total 456287 x-sell out of which 2 canceled
# - There are total 150261 walk-in out of which 0 canceld
# 

# 
# # CHANNEL_TYPE

# - overall 

# In[128]:


overalldata(data,"CHANNEL_TYPE")


# - Approved

# In[129]:


overalldata(dataA,"CHANNEL_TYPE")


# - There are total 719968 Credit and cash offices out of which 289056 Approved
# - There are total  494690 Country-wide    out of which 402787 Approved
# - There are total 212083 Stone out of which 189135 Approved
# - There are total 108528 Regional / Local out of which 96417 Approved
# - There are total 71297 Contact center out of which 25220 Approved
# 
# - There are total  57046 AP+ (Cash loan)    out of which 31231 Approved
# - There are total 6150 Channel of corporate sales out of which 2649 Approved
# - There are total 452 Car dealer out of which 286 Approved
# 
#  
# 

# - Refused

# In[130]:


overalldata(dataR,"CHANNEL_TYPE")


# - There are total 719968 Credit and cash offices out of which 150450 Refused
# - There are total  494690 Country-wide    out of which 65762 Refused
# - There are total 212083 Stone out of which 21988 Refused
# - There are total 108528 Regional / Local out of which 11326 Refused
# - There are total 71297 Contact center out of which 15566 Refused
# 
# - There are total  57046 AP+ (Cash loan)    out of which 22099 Refused
# - There are total 6150 Channel of corporate sales out of which 3365 Refused
# - There are total 452 Car dealer out of which 122 Refused

# - Canceled

# In[131]:


overalldata(dataC,"CHANNEL_TYPE")


# - There are total 719968 Credit and cash offices out of which 279973 canceled
# - There are total  494690 Country-wide    out of which 1782 canceled
# - There are total 212083 Stone out of which 67 canceled
# - There are total 108528 Regional / Local out of which 95 canceled
# - There are total 71297 Contact center out of which 30511 canceled
# 
# - There are total  57046 AP+ (Cash loan)    out of which 3711 canceled
# - There are total 6150 Channel of corporate sales out of which 136 canceled
# - There are total 452 Car dealer out of which 44 canceled

# - Unused

# In[132]:


overalldata(dataU,"CHANNEL_TYPE")


# - There are total 719968 Credit and cash offices out of which 489 canceled
# - There are total  494690 Country-wide    out of which 24359 canceled
# - There are total 212083 Stone out of which 893 canceled
# - There are total 108528 Regional / Local out of which 690 canceled
# - There are total 71297 Contact center out of which 0 canceled
# 
# - There are total  57046 AP+ (Cash loan)    out of which 5 canceled
# - There are total 6150 Channel of corporate sales out of which 0 canceled
# - There are total 452 Car dealer out of which 0 canceled

# # NAME_SELLER_INDUSTRY

# - overall data

# In[133]:


overalldata(data,"NAME_SELLER_INDUSTRY")


# 

# - approved

# In[134]:


overalldata(dataA,"NAME_SELLER_INDUSTRY")


# - There are total 855720 XNA out of which 349962 approved
# - There are total  398265 Consumer electronics    out of which 345194 approved
# - There are total 276029 Connectivity out of which 216284 approved
# - There are total 29781 Construction out of which 26618 approved
# - There are total 23949 Clothing out of which 21611 approved
# - There are total 57849 furniture out of which 51706 approved
# - There are total  57046 Industry   out of which 19194 approved
# - There are total 19194 Auto technology out of which 4515 approved
# - There are total 2709 Jewelry out of which 2465 approved
# - There are total 1215 MLM partners out of which 797 approved
# - There are total 513 Tourism out of which 452  approved

# - rejected

# In[135]:


overalldata(dataR,"NAME_SELLER_INDUSTRY")


# - There are total 855720 XNA out of which 191389 rejected
# - There are total  398265 Consumer electronics    out of which 49510 rejected
# - There are total 276029 Connectivity out of which 35902 rejected
# - There are total 29781 Construction out of which 3076 rejected
# - There are total 23949 Clothing out of which 2265 rejected
# - There are total 57849 furniture out of which 5646 rejected
# - There are total  57046 Industry   out of which 1910 rejected
# - There are total 19194 Auto technology out of which 468 rejected
# - There are total 2709 Jewelry out of which 243 rejected
# - There are total 1215 MLM partners out of which 208 rejected
# - There are total 513 Tourism out of which 61  rejected

# - canceled

# In[136]:


overalldata(dataC,"NAME_SELLER_INDUSTRY")


# - There are total 855720 XNA out of which 313861 canceled
# - There are total  398265 Consumer electronics    out of which 248 canceled
# - There are total 276029 Connectivity out of which 1650 canceled
# - There are total 29781 Construction out of which 11 canceled
# - There are total 23949 Clothing out of which 1 canceled
# - There are total 57849 furniture out of which 286 canceled
# - There are total  57046 Industry   out of which 50 canceled
# - There are total 19194 Auto technology out of which 2 canceled
# - There are total 2709 Jewelry out of which 243 canceled
# - There are total 1215 MLM partners out of which 209 canceled
# - There are total 513 Tourism out of which 1  canceled

# - Unused

# In[137]:


overalldata(dataU,"NAME_SELLER_INDUSTRY")


# - There are total 855720 XNA out of which 508 unused
# - There are total  398265 Consumer electronics    out of which 3313 unused
# - There are total 276029 Connectivity out of which 22193 unused
# - There are total 29781 Construction out of which 76 unused
# - There are total 23949 Clothing out of which 72 unused
# - There are total 57849 furniture out of which 211 unused
# - There are total  57046 Industry   out of which 57 unused
# - There are total 19194 Auto technology out of which 5 unused
# - There are total 2709 Jewelry out of which 243 unused
# - There are total 1215 MLM partners out of which 1 unused
# - There are total 513 Tourism out of which 0  unused
