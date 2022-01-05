import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_palette("dark")
import scipy.stats as stats
from statsmodels.stats.weightstats import ztest as ztest
from statsmodels.stats.proportion import proportions_ztest
pd.set_option('display.max_columns', None)


#Read dataset and create dataframes
df_national = pd.read_csv('../data/2001-2019-National-Data.csv')
df_age = pd.read_csv('../data/2001-2019-National-Data-Age-Sex.csv')

#Display all columns on dataframe
def lst_columns(df):
    df = list(df.columns)
    return df

def filter_column(df, col, value):
    df = df[df[col] == value]
    return df

def plot_stacked_bar_chart(df,col1,col2):
    N = 19
    ind = np.arange(N)  
    width = 0.8
    fig, ax = plt.subplots(figsize =(10, 8), dpi=200)
    ax1 = plt.bar(ind, df[col1], width, color=[input('color 1: ')])
    ax2 = plt.bar(ind, df[col2], width, color=[input('color 2: ')], bottom = df[col1])
    plt.title(input('Fig Title: '), fontsize = 25)
    plt.xlabel('Year', fontsize = 15)
    plt.ylabel(input('Y Label: '), fontsize = 15)
    plt.xticks(ind, (df['Year']))
    plt.yticks(np.arange(start=int(input('yticks start: ')),stop=int(input('yticks stop:')),step=int(input('yticks step:'))))
    plt.legend((ax1[0], ax2[0]), (input('label 1: '), input('label 2: ')), loc='upper left')
    plt.bar_label(ax.containers[0], size=8)
    plt.bar_label(ax.containers[1], size=8)
    fig.tight_layout()
    plt.show()

#Plot Line chart of 2 categories
def plot_line_chart(df,col1,col2):
    fig, ax = plt.subplots(figsize = (10,8), dpi=200)
    sns.lineplot(x = df['Year'], y =df[col1], marker='o', label= input('label 1: '), color=input('color 1: '))
    sns.lineplot(x = df['Year'], y =df[col2], marker='o', label= input('label 2: '), color=input('color 2: '))
    ax.set_title(input('Fig Title: '), fontsize = 25)
    ax.set_ylabel(input('Y Label: '), fontsize = 15)
    ax.set_xlabel('Year', fontsize = 15)
    ax.set_xticks(df['Year'])
    ax.set_yticks(np.arange(start=int(input('yticks start: ')),stop=int(input('yticks stop: ')),step=int(input('yticks step: '))))
    plt.legend(loc='upper left')
    plt.yticks(fontsize = 10)
    fig.tight_layout()
    plt.show()

#Plot Line chart of 4 categories
def plot_line_chart_4(df,col1,col2,col3,col4):
    fig, ax = plt.subplots(figsize = (10,8), dpi=200)
    sns.lineplot(x = df['Year'], y =df[col1], marker='o', label= input('label 1: '), color=input('color 1: '))
    sns.lineplot(x = df['Year'], y =df[col2], marker='o', label= input('label 2: '), color=input('color 2: '))
    sns.lineplot(x = df['Year'], y =df[col3], marker='o', label= input('label 3: '), color=input('color 3: '))
    sns.lineplot(x = df['Year'], y =df[col4], marker='o', label= input('label 4: '), color=input('color 4: '))
    ax.set_title(input('Fig Title: '), fontsize = 25)
    ax.set_ylabel(input('Y Label: '), fontsize = 15)
    ax.set_xlabel('Year', fontsize = 15)
    ax.set_xticks(df['Year'])
    ax.set_yticks(np.arange(start=int(input('yticks start: ')),stop=int(input('yticks stop: ')),step=int(input('yticks step: '))))
    plt.legend(loc='upper left')
    plt.yticks(fontsize = 10)
    fig.tight_layout()
    plt.show()

#Proportions_ztest from statsmodels.stats.proportion
def two_sample_proportions_ztest(sample_success_a, sample_size_a,sample_success_b, sample_size_b):
    alpha = 0.017
    successes = np.array([sample_success_a, sample_success_b])
    samples = np.array([sample_size_a, sample_size_b])
    z_stat, p_value = proportions_ztest(count=successes, nobs=samples, alternative='larger')
    # report
    print('alpha: %0.3f, z_stat: %0.3f, p_value: %0.3f' % (alpha,z_stat, p_value))
    if p_value > alpha:
        print ("Fail to reject the null hypothesis - not significant difference")
    else:
        print ("Reject the null hypothesis -> significant difference")

#Two sample approximate test for population proportions from Hypothesis Testing Lecture
def two_sample_approximate_test_population_proportions(sample_success_a, sample_size_a,sample_success_b, sample_size_b):
    alpha = 0.017
    shared_sample_freq = (sample_success_a + sample_success_b) / (sample_size_a + sample_size_b)
    shared_sample_variance = (sample_size_a + sample_size_b) * (shared_sample_freq * (1 - shared_sample_freq)) / ((sample_size_a * sample_size_b))
    difference_in_proportions = stats.norm(0, np.sqrt(shared_sample_variance))
    vet_sample_freq = sample_success_a/sample_size_a
    pop_sample_freq = sample_success_b/sample_size_b
    difference_in_sample_proportions = vet_sample_freq - pop_sample_freq
    p_value = 1 - difference_in_proportions.cdf(difference_in_sample_proportions)
        # report
    print('alpha: %0f, difference_in_sample_proportions: %0f, p_value: %0f' % (alpha,difference_in_sample_proportions, p_value))
    if p_value > alpha:
        print ("Fail to reject the null hypothesis - not significant difference")
    else:
        print ("Reject the null hypothesis -> significant difference")

if __name__ == "__main__":

    df_age = df_age[df_age['Age Group'] != 'Total']

#All columns from df_national
['Year',
 'Veteran Suicide Deaths',
 'Veteran Population Estimate',
 'Veteran Crude Rate per 100,000',
 'Veteran Age Adjusted Rate per 100,000',
 'Veteran Age and Sex Adjusted Rate per 100,000',
 'Male Veteran Suicide Deaths',
 'Male Veteran Population Estimate',
 'Male Veteran Crude Rate per 100,000',
 'Male Veteran Age Adjusted Rate per 100,000',
 'Female Veteran Suicide Deaths',
 'Female Veteran Population Estimate',
 'Female Veteran Crude Rate per 100,000',
 'Female Veteran Age Adjusted Rate per 100,000',
 'Non-Veteran Suicide Deaths',
 'Non-Veteran Population Estimate',
 'Non-Veteran Crude Rate per 100,000',
 'Non-Veteran Age Adjusted Rate per 100,000',
 'Non-Veteran Age and Sex Adjusted Rate per 100,000',
 'Male Non-Veteran Suicide Deaths',
 'Male Non-Veteran Population Estimate',
 'Male Non-Veteran Crude Rate per 100,000',
 'Male Non-Veteran Age Adjusted Rate per 100,000',
 'Female Non-Veteran Suicide Deaths',
 'Female Non-Veteran Population Estimate',
 'Female Non-Veteran Crude Rate per 100,000',
 'Female Non-Veteran Age Adjusted Rate per 100,000',
 'US Population Suicide Deaths',
 'US Population Population Estimate',
 'US Population Crude Rate per 100,000',
 'US Population Age Adjusted Rate per 100,000',
 'US Population Age and Sex Adjusted Rate per 100,000',
 'Male US Population Suicide Deaths',
 'Male US Population Population Estimate',
 'Male US Population Crude Rate per 100,000',
 'Male US Population Age Adjusted Rate per 100,000',
 'Female US Population Suicide Deaths',
 'Female US Population Population Estimate',
 'Female US Population Crude Rate per 100,000',
 'Female US Population Age Adjusted Rate per 100,000']

#All columns from df_national
['Year',
 'Age Group',
 'Veteran Suicide Deaths',
 'Veteran Population Estimate',
 'Veteran Crude Rate per 100,000',
 'Male Veteran Suicide Deaths',
 'Male Veteran Population Estimate',
 'Male Veteran Crude Rate per 100,000',
 'Age Group 2',
 'Female Veteran Suicide Deaths',
 'Female Veteran Population Estimate',
 'Female Veteran Crude Rate per 100,000',
 'Veteran Population Estimate in millions']

#Columns to keep
lst_keep_column = ['Occur Date', 'UCR Literal']