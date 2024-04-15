# Association-Rule
Association rule 
# Association Rule Mining
## Problem Statement:
#### The famous bookstore in India, Kitabi Duniya, has been experiencing a decline in growth due to online book selling and widespread Internet access.
# `CRISP-ML(Q)` process model describes six phases:

# 1. Business and Data Understanding
# 2. Data Preparation
# 3. Model Building
# 4. Model Evaluation
# 5. Deployment
# 6. Monitoring and Maintenance
'''
**Objective(s):** maximize footfalls of customers 

**constraints** : maximize profits

Success Criteria

# Business Success Criteria:  Increase sales by 25%
# ML Success Criteria: machine learning model (aprori algorithm ) should predict atleast 95% accuracy
# Economic Success Criteria: Book store   will see an increase in revenues by atleast 30%

# **Proposed Plan:**
# we will be using the Association Rules Algorithm, a pattern mining technique, to identify rules (patterns) that can help improve sales..



#Data Description:
#Data is in the form of binary variable indicating whether the customer purchased a book in the category,1 if customer purchased children's books, 0 otherwise 

ChildBks: "Children's books" 

YouthBks: "Youth books" 

CookBks: "Cooking books" 

DoItYBks: "Do-it-yourself books"

RefBks:"Reference books" 

ArtBks: "Art books" 

GeogBks: "Geography books" 

ItalCook: "Italian cooking books" 

ItalAtlas: "Italian atlas books" 

ItalArt: "Italian art books" 

Florence: "Books about Florence, Italy"
'''
#Importing required libraries
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
# Mlxtend (machine learning extensions) is a Python library of useful tools for the day-to-day data science tasks.

import csv
  
import warnings
warnings.filterwarnings("ignore")
#Reading the data
books_data=pd.read_csv(r"C:\Users\hp\Downloads\Machine larning\Assignment\Association Rule\Association Rule\book.csv")
#Database connection
from sqlalchemy import create_engine
# Credentials to connect to Database
user = 'root'  # user name
pw = 'user1'  # password
db = 'ds'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
# to_sql() - function to push the dataframe onto a SQL table.

books_data.to_sql('books', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

sql = 'select * from books;'
df = pd.read_sql_query(sql, engine)
df.info()
### Elementary Analysis ###
# Most popular items
count = df.loc[:, :].sum()
print(count)

pop_item = count.sort_values(ascending=False).head(10)

pop_item
pop_item = pop_item.to_frame()
pop_item
pop_item = pop_item.reset_index()
pop_item

pop_item = pop_item.rename(columns = {"index": "items", 0: "count"})
pop_item

# Data Visualization
#%matplotlib inline
plt.rcParams['figure.figsize'] = (10, 6) # rc stands for runtime configuration 
matplotlib.style.use('dark_background')
ax = pop_item.plot.barh(x = 'items', y = 'count')
plt.title('Most popular items')
plt.gca().invert_yaxis() # gca means "get current axes"
# Itemsets
frequent_itemsets = apriori(df, min_support = 0.0075, max_len = 4, use_colnames = True)
frequent_itemsets

# Most frequent itemsets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)
frequent_itemsets
# Association Rules
rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
print(rules.head(20))

rules.sort_values('lift', ascending = False).head(10)
### Handling Profusion of Rules (Duplication elimination)

def to_list(i):
    return (sorted(list(i)))
ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X
ma_X = ma_X.apply(sorted)
ma_X

rules_sets = list(ma_X)
rules_sets
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
unique_rules_sets
index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))
    
index_rules
# Rules without any redudancy 
rules_no_redundancy = rules.iloc[index_rules, :]
rules_no_redundancy
# Sorted list and top 10 rules 
rules10 = rules_no_redundancy.sort_values('lift', ascending = False).head(10)

rules10

type(rules10)
rules10.plot(x = "support", y = "confidence", c = rules10.lift, kind="scatter", s = 12, cmap = plt.cm.coolwarm)
