import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

TEST_FILE = "data/test.csv"
TRAIN_FILE = "data/train.csv"
RESULT_FILE = "data/contest_submission.csv"

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def load_file(file):
    return pd.read_csv(file)


def write_file(file, dataframe):
    pd.DataFrame.to_csv(dataframe, file, index=False)


def load_test_file():
    m_pd_test = load_file(TEST_FILE)[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare',
                                      'Cabin', 'Embarked']]
    print(m_pd_test['Sex'].unique())
    m_pd_test.loc[m_pd_test['Sex'] == 'male', 'Sex'] = 0
    m_pd_test.loc[m_pd_test['Sex'] == 'female', 'Sex'] = 1
    m_pd_test = m_pd_test.astype({'Sex': 'int64'})
    age_mean = round(m_pd_test['Age'].mean(), 2)
    m_pd_test.loc[m_pd_test['Age'].isnull(), 'Age'] = age_mean
    age_fare = round(m_pd_test['Fare'].mean(), 2)
    m_pd_test.loc[m_pd_test['Fare'].isnull(), 'Fare'] = age_fare
    m_pd_test.columns = pd.MultiIndex.from_tuples(
        list(zip(['features'] * 11,
                 ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin',
                  'Embarked']))) \
        .drop_duplicates()
    print(m_pd_test.head())
    print(m_pd_test.describe())
    return m_pd_test


def load_train_file():
    m_pd_train = load_file(TRAIN_FILE)[
        ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare',
         'Cabin', 'Embarked', 'Survived']]
    print(m_pd_train['Sex'].unique())
    m_pd_train.loc[m_pd_train['Sex'] == 'male', 'Sex'] = 0
    m_pd_train.loc[m_pd_train['Sex'] == 'female', 'Sex'] = 1
    m_pd_train = m_pd_train.astype({'Sex': 'int64'})
    age_mean = round(m_pd_train['Age'].mean(), 2)
    m_pd_train.loc[m_pd_train['Age'].isnull(), 'Age'] = age_mean
    m_pd_train.columns = pd.MultiIndex.from_tuples(
        list(zip(['features'] * 11,
                 ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin',
                  'Embarked'])) + [('label', 'Survived')]) \
        .drop_duplicates()
    print(m_pd_train.head())
    print(m_pd_train.describe())
    return m_pd_train


def generate_corr(pd_train_p):
    print("**************** CORR. MATRIX ***************+++")
    print(pd_train_p.corr())
    print("**************** CORR. MATRIX ***************+++")


def train_model(pd_train_p, pd_test_p):
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    X = pd_train_p['features']
    y = pd_train_p[('label', 'Survived')]
    model.fit(X, y)
    predictions = model.predict(pd_test_p['features'])
    pd_result = pd.DataFrame()
    pd_result['PassengerId'] = pd_test_p[('features', 'PassengerId')].ravel()
    pd_result['Survived'] = predictions
    return pd_result


def generate_report(pd):
    report = pandas_profiling.ProfileReport(df)

if __name__ == '__main__':
    pd_train = load_train_file()
    pd_train_final = pd_train[[('features', 'PassengerId'), ('features', 'Pclass'), ('features', 'Sex'),
                               ('features', 'Age'), ('features', 'SibSp'), ('features', 'Parch'), ('features', 'Fare'),
                               ('label', 'Survived')]]
    generate_corr(pd_train)
    pd_test = load_test_file()
    pd_test_final = pd_test[[('features', 'PassengerId'), ('features', 'Pclass'), ('features', 'Sex'),
                             ('features', 'Age'), ('features', 'SibSp'), ('features', 'Parch'), ('features', 'Fare')]]
    ret = train_model(pd_train_final, pd_test_final)
    print(ret)
    write_file(RESULT_FILE, ret)
