#pip install pandas
import pandas as pd
import numpy as np
"""
#data=pd.Series(data,index,dtype=None,copy=False)
data=pd.Series()#empty series
print(data)

data=np.array([1,2,3,4,5])
print(type(data))
s=pd.Series(data,index=['a','b','c','d','e'])
print(s)
print(type(s))

print(s[2])
print("The value is:",s[['a','c','e']])

data={'a':10,'b':20,'c':30,'d':40}
s=pd.Series(data)
print(s)
print(s[0])

s=pd.Series(10,index=[100,200,300,400,500])
print(s)
print(s[100])

"""
#data=pd.DataFrame(data,index,columns,dtype,copy)
"""
data=pd.DataFrame()
print(data)

data=[1,2,3,4,5,6]
df=pd.DataFrame(data,index=['a','b','c','d','e','f'],columns=['one'])
print(df)
"""

"""
data={'Name':['a','b','c','d','e','f'],'Age':[20,10,30,40,20,10]}
df=pd.DataFrame(data,index=['a1','b1','c1','d1','e1','f1'])
print(df)
print("The Column Value is:",df['Age'])
df['Marks']=pd.Series([100,200,300,400,500,600],index=['a1','b1','c1','d1','e1','f1'])
print(df)

df1=pd.DataFrame(df,index=['a1','b1','c1','d1','e1','f1'])
print(df1)
"""
"""
data={'Name':['a','b','c','d','e','f'],'Age':[20,10,30,40,20,10]}
df=pd.DataFrame(data,index=['a1','b1','c1','d1','e1','f1'])
print(df)
df.loc['g1']='h',50
print(df)

df.loc['i1']=df.loc['f1']+df.loc['g1']
print(df)

df=df.drop('d1')
print(df)
"""
"""
df['Total']=df['Age']+df['Marks']
print(df)

del df['Age']
print(df)

df.pop('Marks')
print(df)"""
#pip install matplotlib
import matplotlib.pyplot as plt
df= pd.read_csv("Internet Users.csv")
print(df.head())
print(df['Country'])

print(df.describe())

print(np.mean(df['InternetUsers']))
print(np.median(df["InternetUsers"]))
print(np.sum(df[['InternetUsers','Population']]))
print(df.isnull().any())
print(df.notnull().any())

#df=df.fillna(method='pad')#forward fill 
#df=df.fillna(method='bfill')#backward fill
#df=df.fillna(0)#filling with scaler values
df=df.dropna()#drop na
print(df)

plt.scatter(df["Country"],df['InternetUsers'])
plt.xlabel="Country"
plt.ylabel="InternetUsers"
plt.show()

plt.bar(df["Country"],df['InternetUsers'])
plt.xlabel="Country"
plt.ylabel="InternetUsers"
plt.show()

plt.barh(df["Country"],df['InternetUsers'])
plt.xlabel="Country"
plt.ylabel="InternetUsers"
plt.show()


mylabels=["Brazil","China","India","Japan","UnitedStates"]
plt.pie(df['InternetUsers'],labels=mylabels)
plt.axis('equal')
plt.show()
"""
plt.subplot(1,2,1)
plt.bar(df["Country"],df['InternetUsers'])
plt.xlabel="Country"
plt.ylabel="InternetUsers"
plt.title("InternetUsers")


plt.subplot(1,2,2)
plt.bar(df["Country"],df['Population'])
plt.xlabel="Country"
plt.ylabel="Population"
plt.title("Population")

plt.suptitle("Country based result")
plt.show()

"""



"""
data={'Name':pd.Series(['a','b','c','d','e'],index=[100,200,300,400,500]),'Age':pd.Series([20,10,30,40,20,10],index=[100,200,300,400,500,600])}
df=pd.DataFrame(data)
print(df)
"""
"""
data=pd.read_csv("D:/Desktop/desktop files/NFHS4-MP_Districts_All_1_1.csv")
#print(data)
print(data.head(7))
print(data.tail(3))
print(data.describe())
print(data.isnull().any())
print(data.notnull().any())
"""
"""
data=pd.read_csv("D:/Desktop/desktop files/New folder/iris.csv")
print(data)
print(data.head(7))

print(data.tail(7))

print(data.describe())

print(data.isnull().any())
print(data.notnull().any())

print(data[['sepal.width','sepal.length']])
print(np.sum(data['petal.width']))

#print(data['sepal.width'].head())
"""
"""
data['sum']=data['sepal.width']+data['petal.width']

print(data['sum'])

data['status'] = np.where(data['sum'] <4 , 'null', data['sum'])
print(data['status'])

data['status']=np.where(data[percent]<60,'fail','pass')

import matplotlib.pyplot as plt

plt.bar(data['sum'],data['status'],color='green')

data.head(7).plot.bar()
data.head(7).plot.bar(stacked=True)"""
"""
"""
"""
data={'Name':pd.Series(['a','b','c','d','e'],index=[100,200,300,400,500]),
      'Age':pd.Series([20,10,30,40,20,10],index=[100,200,300,400,500,600])}
df=pd.DataFrame(data)
print(df)


data1={'Name':['a','b','c','d','e','f'],'Age':[20,10,30,40,20,10]}
df1=pd.DataFrame(data1,index=['a','b','c','d','e','f'])
print(df1)

#data3= pd.concat([df,df1])
#print(data3)

#pd.merge(data,data1,how=inner,on)

data2=pd.merge(df,df1,on='Name')
print(data2)

data2=pd.merge(df,df1,how='left',on='Name')
print(data2)

data2=pd.merge(df,df1,how='right',on='Name')
data2.columns=['Name',"Age1","Age2"]
print(data2)"""

"""
import datetime as dt

today=dt.datetime.now()
print(today)

today=dt.date.today()
print(today)


previous=today-dt.timedelta(days=2)
print(previous)

advance=today+dt.timedelta(days=5)
print(advance)

print(pd.date_range('11/1/2019', periods=20,freq='A-JAN'))"""
"""
D=day 
M=Month end
MS= Month start
SM= Semi month
A=annual
BAS= business year start
"""

import matplotlib.pyplot as plt
"""
mu=0.5
sigma=0.1
s=np.random.normal(mu,sigma,100)
count,bins,ignore=plt.hist(s,20,normed=True)
plt.plot(bins,1/(sigma*np.sqrt(2*np.pi))*np.exp(-(bins-mu)**2/(2*sigma**2))),
plt.show()

"""







"""
from scipy.stats import binom,bernoulli,poisson
import seaborn as sb

#binom.rvs(size=10,n=20,p=0.8)
data=binom.rvs(n=20,p=0.8,size=1000)
ax=sb.distplot(data,kde=True,color='blue',hist_kws={'linewidth':25,'alpha':1})
ax.set(xlabel='binomial',ylabel="frequency")"""

"""
data=bernoulli.rvs(size=1000,p=0.6)
ax=sb.distplot(data,kde=True,color='crimson',hist_kws={'linewidth':25,'alpha':1})
ax.set(xlabel='bernoulli',ylabel="frequency")"""

"""

data=poisson.rvs(mu=4, size=10000)
ax=sb.distplot(data,kde=True,color='green',hist_kws={'linewidth':25,'alpha':1})
ax.set(xlabel='poisson',ylabel="frequency")
"""






"""
df=sb.load_dataset('iris')
print(df)
print(df.describe())

sb.pairplot(df,hue='species')
plt.show()
"""

"""

from scipy import stats
x=np.linspace(0,10,100)
fig,ax=plt.subplots(1,1)
linestyles=[":","--","-","-."]
dof=[1,4,6,7]
for df,ls in zip(dof,linestyles):
    ax.plot(x,stats.chi2.pdf(x,df),linestyle=ls)

plt.xlim(0,10)
plt.ylim(0,0.4)
plt.legend([1,4,6,7])
plt.show()
"""
"""
x=np.arange(0,10)
y=x^2
z=x^3
t=x^4
#y=2*x
plt.xlabel("value of x")
plt.ylabel("value of y")
plt.title("Graph Plotting")
plt.plot(x,y,'r')
plt.plot(x,y,'<')
plt.plot(x,y,linestyle='--')
#plt.plot(x,y,color='black')
plt.annotate(xy=[2,0],s="highest value")

#plt.savefig('D:/graph.pdf',format='pdf')"""
#plt.plot(x,z)
#plt.plot(x,t)
#plt.legend(['Value1','Value2','Value3'])

#plt.show()
"""
data={'Name':['add','bcc','cad','ddd','esd','fdfd'],'s1':[20,10,30,40,20,10],'s2':[12,20,30,40,50,30],'s3':[13,14,15,16,17,50]}
df=pd.DataFrame(data,index=['a','b','c','d','e','f'])
df["total"]=df["s1"]+df["s2"]+df["s3"]
print(df)

"""






