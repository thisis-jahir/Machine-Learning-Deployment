#######Machine Learning Deployment######
##Libraries
import tkinter
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from tkinter.filedialog import askopenfilename
##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
##
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
#--------------------xxx--------------------------------
###Windows creation
class window:
    #browse dataset
    def window1(self):
        window1 = Tk()
        window1.title('ML Algorithms')
        window1.config(bg='lightblue')
        window1.state('zoomed')
        
        label1 = Label(window1,text='Machine Learning Algorithms',bg='lightblue',font=('helvetica',18,'bold')).place(x=520,y=150)
        label2 = Label(window1,bg='lightyellow',relief='solid').place(x=500,y=300,height=200,width=400)
        label3 = Label(window1,text='Upload your Dataset',bg='lightyellow',font=('helvetica',12,'bold')).place(x=615,y=320)
        
        def browse():
            global file,data
            file = filedialog.askopenfilename(initialdir="F:/",filetypes = [("csv files","*.csv")])
            entry1.insert(0,file)
            data = pd.read_csv(file)
        

        entry1 = Entry(window1)
        entry1.place(x=543,y=380,height=40,width=320)
        entry1.configure(bg='lightblue')
        
        button1 = Button(window1,bg='lightblue',text='Browse',command= browse).place(x=765,y=380,height=40,width=100)
        button2 = Button(window1,bg='lightblue',text='Next',command= w.window2).place(x=645,y=440,height=30,width=100)
        

        window1.mainloop()
      
        
    ##Confirm dataset
    def window2(self):
        window2 = Tk()
        window2.title('ML Algorithms')
        window2.configure(bg='lightblue')
        window2.state('zoomed')
        
        label1 = Label(window2,text='Confirm your Dataset',bg='lightblue',font=('helvetica',18,'bold'))
        label1.pack(pady=120)
        
        frame = Frame(window2)
        frame.pack(padx=20)
        
        scroll1 = ttk.Scrollbar(frame,orient='vertical')
        scroll1.pack(side=RIGHT,fill=Y)
        
        scroll2 = ttk.Scrollbar(frame,orient='horizontal')
        scroll2.pack(side=BOTTOM,fill=X)

        global tview
        tview = ttk.Treeview(frame, yscrollcommand=scroll1.set, xscrollcommand=scroll2.set)
        tview.pack()

        scroll1.config(command=tview.yview)
        scroll2.config(command=tview.xview)
  
        tview["column"]=list(data.columns)
        tview["show"]="headings"
        
        for column in tview["column"]:
            tview.heading(column, text=column)
        df_rows = data.to_numpy().tolist()
        for row in df_rows:
            tview.insert("","end",values=row)
        tview.pack()
        
        button1 = Button(window2,text='Open',command=w.window3)
        button1.place(x=860,y=570,height=30,width=100)
        button2 = Button(window2,text='Back',command=w.window1)
        button2.place(x=980,y=570,height=30,width=100)
        
        window2.mainloop()
    ##Selecting features
    def window3(self):
        window3=Tk()
        window3.configure(background="lightblue")
        window3.title('ML Algorithms')
        window3.state('zoomed')
      
        label1 = Label(window3,relief='solid')
        label1.place(x=730,y=100,width=600,height=500)
        
        label2 = Label(window3, text="Selected Features", font=("Helvetica", 14, 'bold'))
        label2.pack()
        label2.place(x=930,y=120)

        label3 = Label(window3, text="Select\nIndependent Variables", fg='#000000', bg="lightblue" ,font=("helvetica", 14, 'bold'))
        label3.place(x=70,y=100,width=220,height=50)

        label4 = Label(window3, text="Select\nDependent Variables", fg='#000000', bg="lightblue", font=("helvetica", 14, 'bold'))
        label4.place(x=420,y=100,width=220,height=50)
        
        frame1 = Frame(window3)
        frame1.place(x=30,y=180,width=300,height=400)

        scrollbar1 = Scrollbar(frame1)
        scrollbar1.pack(side=RIGHT,fill=BOTH )

        listbox1 = Listbox(frame1, selectmode=MULTIPLE)
        listbox1.config(yscrollcommand=scrollbar1.set)
        listbox1.place(width=300, height=400)

        scrollbar1.config(command=listbox1.yview)

        j=0
        for i in tview["column"]:
            listbox1.insert(j, i)
            j = j + 1

        frame2 = Frame(window3)
        frame2.place(x=370, y=180, width=300, height=400)

        scrollbar2 = Scrollbar(frame2)
        scrollbar2.pack(side=RIGHT, fill=BOTH)

        listbox2 = Listbox(frame2, selectmode=SINGLE)
        listbox2.config(yscrollcommand=scrollbar2.set)
        listbox2.place(width=300, height=400)

        scrollbar2.config(command=listbox2.yview)

        j = 0
        for i in tview["column"]:
            listbox2.insert(j, i)
            j = j + 1


        def indep_select():
            global indep, indep1
            indep = []
            indep1 = []
            label34 = Label(window3, text="Independent Variables",font=('helvetica',14,'bold'))
            label34.place(x=770, y=170)
            clicked = listbox1.curselection()
            z = 220
            for item in clicked:
                label10 = Label(window3, text=listbox1.get(item),font=('helvetica',9))
                label10.place(x=770, y=z)
                indep.append(item)
                indep1.append(listbox1.get(item))
                z = z + 25


        button1 = Button(window3,text="Choose",command=indep_select,font=("Arial Black", 9, 'bold'))
        button1.place(x=125,y=600,width=100, height=40)

        def dep_select():
            global dep,dep1
            dep = []
            dep1 = []
            label34 = Label(window3, text="Dependent Variables ",font=('helvetica',14,'bold'))
            label34.place(x=1080, y=170)
            clicked = listbox2.curselection()
            z = 220
            for item in clicked:
                label10 = Label(window3, text=listbox2.get(item),font=('helvetica',9))
                label10.place(x=1080, y=z)
                dep.append(item)
                dep1.append(listbox2.get(item))
                z = z + 25


        button2 = Button(window3, text="Choose", command=dep_select,font=("Arial Black", 9, 'bold'))
        button2.place(x=455, y=600, width=100, height=40)
        
        button3 = Button(window3, text = "Confirm",command=w.window4,font=("Arial Black", 9, 'bold'))
        button3.place(x=840, y=620, height=50, width=180)
        
        button4 = Button(window3, text = "Back",font=("Arial Black", 9, 'bold'))
        button4.place(x=1050, y=620, height=50, width=180)
        button4.configure(command=w.window2)
        
        window3.mainloop()
    
    ##Reg,class,cluster
    def window4(self):
        window4 = Tk()
        window4.configure(bg="lightblue")
        window4.title("ML Algorithms")
        window4.state('zoomed')

        label1 = Label(window4, text="Select the Problem Type",bg='lightblue', font=("helvetica", 16, 'bold'))
        label1.place(x=580,y=200)
        
        label2 = Label(window4,relief='solid')
        label2.place(x=300,y=300,height=160,width=800)
        
        button1 = Button(window4, text="Regression", bg="lightblue",command=w.regression)
        button1.place(x=340,y=360,width=130,height=40)
        
        button2 = Button(window4, text="Classification", bg="lightblue",command=w.classification)
        button2.place(x=640,y=360,width=130,height=40)
        
        button3 = Button(window4, text="Clustering", bg="lightblue", command=w.clustering)
        button3.place(x=930,y=360,width=130,height=40)

        window4.mainloop()
        
    ##Regression Algorithms
    def regression(self):
        window5 = Tk()
        window5.configure(bg="lightblue")
        window5.title("ML - Regression Algorithms")
        window5.state('zoomed')
        
        
        global algos
       
        algos="Regression"

        label1 = Label(window5, text="Select the Alogirithm",bg="lightblue", font=("helvetica",18,"bold"))
        label1.place(x=560,y=180)
        
        label2 = Label(window5,relief='solid')
        label2.place(x=435,y=260,height=200,width=500)
        
        

        def comboclick(event):
            x = data.iloc[:, indep].values  
            y = data.iloc[:, dep].values 
            global x_train,y_train
            
            global train_accuracy,test_accuracy,regressor,entryp
            
            if (combo.get() == "Simple Linear Regression"):                                
                x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
                # regressor=LinearRegression()
                # regressor.fit(x_train,np.ravel(y_train))
                # train_accuracy=regressor.score(x_train,y_train)
                # test_accuracy=regressor.score(x_test,y_test)
                

            elif (combo.get() == "Multiple Linear Regression"):
               

                # Splitting the dataset into training and test set.
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
                # regressor = LinearRegression()
                # regressor.fit(x_train, y_train)                
                # train_accuracy = regressor.score(x_train, y_train)
                # tesst_accuracy = regressor.score(x_test, y_test)
               

            elif (combo.get() =="Polynomial Regression"):    
                
                # Training the dataset in polynomial regression
                # global poly_reg
                
                label = Label(window5,text='Enter Polynomial degree',bg='lightblue')
                label.place(x=550,y=350,heigh=30)
                
                entryp = Entry(window5,text='poly degree',bg='lightblue')
                entryp.place(x=700,y=350,heigh=30)
                
                
                # poly_reg = PolynomialFeatures(degree=5)
                # x_poly = poly_reg.fit_transform(x)
               
                # regressor = LinearRegression()
                # regressor.fit(x_poly,np.ravel(y))              
                # train_accuracy = regressor.score(x_poly,y)
               

            elif(combo.get()=="SVM Regression"):              
                global sc_x,sc_y
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

                sc_x = StandardScaler()
                sc_y = StandardScaler()
                x = sc_x.fit_transform(x)
                y = sc_y.fit_transform(y)

                regressor = SVR(kernel='rbf')
                regressor.fit(x_train,y_train)
                y_pred=regressor.predict(x_test)

                train_accuracy = regressor.score(x_train,y_train)
                test_accuracy = regressor.score(x_test,y_test)
               
            elif(combo.get()=="Decision Tree Regression"): 
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

                regressor = DecisionTreeRegressor(random_state=0)
                regressor.fit(x_train,y_train)
                y_pred=regressor.predict(x_test)

                train_accuracy = regressor.score(x_train,y_train)
                test_accuracy = regressor.score(x_test,y_test)
               
            elif(combo.get()=="Random Forest Regression"):  
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

                regressor = RandomForestRegressor(n_estimators=10,random_state=0)
                regressor.fit(x_train,y_train)
                train_accuracy = regressor.score(x_train,y_train)
                test_accuracy = regressor.score(x_test,y_test)
                y_pred=regressor.predict(x_test)
                
            else:
                pred = regressor.predict([result])
                label64 = Label(window6,text="Click next to view the Prediction")
                label64.place(x=260,y=z+100)

        options = ["Simple Linear Regression", "Multiple Linear Regression", "Polynomial Regression","SVM Regression","Decision Tree Regression","Random Forest Regression"]
        global combo
        combo = ttk.Combobox(window5, value=options)
        combo.config(width=50)
        combo.current(0)
        combo.bind("<<ComboboxSelected>>", comboclick)
        combo.pack()
        combo.place(x=535,y=300,height=40,width=300)

        button6 = Button(window5, text='Train Model', command=w.window6)
        button6.pack()
        button6.config(bg='lightblue')
        button6.place(x=630,y=385,height=50,width=100)

        window5.mainloop()

    ##Classification Algorithms
    def classification(self):
        window8 = Tk()
        window8.configure(bg="lightblue")
        window8.state('zoomed')
        window8.title("ML- Classification Algorithms")

        global algos
        algos="Classification"

        label3 = Label(window8, text="Select the Algorithm", bg="lightblue", font=("helvetica",18,"bold"))
        label3.place(x=560,y=180)
        
        label2 = Label(window8,relief='solid')
        label2.place(x=435,y=260,height=200,width=500)
        
        def comboclick1(event):
            
            x = data.iloc[:, indep].values
            y = data.iloc[:, dep].values
            global x_train,y_train
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

            global sc
            sc = StandardScaler()
            x_train = sc.fit_transform(x_train)
            x_test = sc.transform(x_test)

            if(combo2.get()=="Logistic Regression"): 
                print('logistic regression')
                # global classifier
                # classifier = LogisticRegression(random_state=0)
                # classifier.fit(x_train,y_train)

                # y_pred = classifier.predict(x_test)

                # cm = confusion_matrix(y_test,y_pred)
                # global acc
                # acc = (sum(np.diag(cm))/len(y_test))

            elif(combo2.get()=="Naive Bayes Classification"):            
                classifier = GaussianNB()
                classifier.fit(x_train ,y_train)

                y_pred = classifier.predict(x_test)

                cm = confusion_matrix(y_test,y_pred)
                acc = (sum(np.diag(cm))/len(y_test))

            elif(combo2.get()=="K-Nearest Neighbour Classification"):                
                classifier = KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2)
                classifier.fit(x_train,y_train)

                y_pred = classifier.predict(x_test)

                cm = confusion_matrix(y_test,y_pred)
                acc = sum(np.diag(cm))/len(y_test)

            elif(combo2.get()=="SVM Classification"):              
                classifier = SVC(kernel='rbf',random_state=0)
                classifier.fit(X_train,y_train)

                y_pred=classifier.predict(X_test)

                cm = confusion_matrix(y_test,y_pred)
                acc = sum(np.diag(cm))/len(y_test)

            elif(combo2.get()=="Decision Tree Classification"):
                classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
                classifier.fit(x_train,y_train)

                y_pred = classifier.predict(x_test)

                cm = confusion_matrix(y_test,y_pred)
                acc = sum(np.diag(cm))/len(y_test)

            elif(combo2.get()=="Random Forest Classification"):
                classifier = RandomForestClassifier()
                classifier.fit(x_train,y_train)

                y_pred = classifier.predict(x_test)
                cm = confusion_matrix(y_test,y_pred)
                acc = sum(np.diag(cm))/len(y_test)

        options1 = ["Logistic Regression","Naive Bayes Classification","K-Nearest Neighbour Classification","SVM Classification","Decision Tree Classification","Random Forest Classification"]
        global combo2
        combo2 = ttk.Combobox(window8, value=options1)
        combo2.config(width=50)
        combo2.current(0)
        combo2.bind("<<ComboboxSelected>>", comboclick1)
        combo2.pack()
        combo2.place(x=535,y=300,height=40,width=300)

        button6 = Button(window8, text='Train Model',command=w.window6)
        button6.pack()
        button6.place(x=630,y=385,height=50,width=100)


        window8.mainloop()
    
    ##Clustering Algorithms
    def clustering(self):
        window9 = Tk()
        window9.configure(bg="lightblue")
        window9.state('zoomed')
        window9.title("ML-Clustering Algorithm")
        
        global algos
        algos = "clustering"

        label1 = Label(window9, text="Select the Algorithm", bg="lightblue", font=("helvetica", 18,"bold"))
        label1.place(x=560,y=180)
        
        label2 = Label(window9,relief='solid')
        label2.place(x=435,y=260,height=200,width=500)

        def comboclick2(event):
            X = data.iloc[:, indep].values
            global clusterkmeans,clustersch
            if(combo3.get()=="KMeans Clustering"):

                wcss=[]

                for i in range(1,11):
                    kmeans = KMeans(n_clusters=i,random_state=0)
                    kmeans.fit(X)
                    wcss.append(kmeans.inertia_)
                def plot():
                   
                    plt.figure(figsize = (8,5), dpi=50)
                    plt.scatter(range(1,11),wcss)
                    plt.plot(range(1,11),wcss)
                    plt.title("Elbow Method")
                    plt.xlabel("Number of Clusters")
                    plt.ylabel("WCSS")
                    plt.show()

                button91 = Button(window9, text="KMeans - plot",bg='lightblue', command=plot)
                button91.place(x=500,y=350,width=120)
                
                label = Label(window9,text='Cluster size',bg='lightblue')
                label.place(x=630,y=350,height=30,width=100)
                
                clusterkmeans = Entry(window9,bg='lightblue')
                clusterkmeans.place(x=750,y=350,height=30)

               
            elif(combo3.get()=="Hierarchical Clustering"):
                

                def plot1():
                    plt.figure(figsize=(8,5),dpi=50)
                    dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))
                    plt.title("Dendrogram")
                    plt.xlabel("X-Values")
                    plt.ylabel("Euclidean Distance")
                    plt.show()

                
                button = Button(window9, text="Dendrogram - plot",bg='lightblue', command=plot1)
                button.place(x=500,y=350,width=120)
                
                labell = Label(window9,text='Cluster size',bg='lightblue')
                labell.place(x=630,y=350,height=30,width=100)
                
                clustersch = Entry(window9,bg='lightblue')
                clustersch.place(x=750,y=350,height=30)

        options2 = ["Select Clustering Algorithm","KMeans Clustering","Hierarchical Clustering"]
        global combo3
        combo3 = ttk.Combobox(window9, value=options2)
        combo3.config(width=50)
        combo3.current(0)
        combo3.bind("<<ComboboxSelected>>", comboclick2)
        combo3.pack()
        combo3.place(x=535,y=300,height=40,width=300)
        
        button6 = Button(window9, text='Train Model', command=w.window7)
        button6.pack()
        button6.config(bg='lightblue')
        button6.place(x=630,y=385,height=50,width=100)



        window9.mainloop()
    #PREDICTION
    def window6(self):
        window6 = Tk()
        window6.configure(bg="lightblue")
        window6.state('zoomed')
        window6.title("ML-Algorithms")
    
        label61 = Label(window6, text="Index Name",  bg="lightblue", font=("helvetica", 20, 'bold'))
        label61.place(x=150, y=100, height=30)
    
        z=140
        global entry61, entries
        entries = []
        for i in indep1:
            label63 = Label(window6, text=i,bg="lightblue", font=("helvetica", 15, 'bold'))
            label63.place(x=145, y=z)
    
            entry61 = Entry(window6)
            entry61.place(x=420, y=z, height=30, width=150)
            entries.append(entry61)
            z = z + 25
    
        label62 = Label(window6, text="Index Values", bg="lightblue", font=("helvetica", 20, 'bold'))
        label62.place(x=415, y=90)
    
        def predict():
            global result
            result=[]
            for entry in entries:
                result.append(entry.get())
            
            if(algos=="Regression"):
                global pred

                if(combo.get()=="Simple Linear Regression"):
                    x = data.iloc[:, indep].values  
                    y = data.iloc[:, dep].values

                    
                    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
                    regressor=LinearRegression()
                    regressor.fit(x_train,np.ravel(y_train))
                    train_accuracy=regressor.score(x_train,y_train)
                    test_accuracy=regressor.score(x_test,y_test)
                    
                    pred = regressor.predict([result])
                    Label(window6,text=str(pred),font=('Helvetica',10,'bold'),bg='light blue', relief='solid').place(x=880,y=580)
                    Label(window6,text=f'Train accuracy : {train_accuracy}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=150)
                    Label(window6,text=f'Test accuracy : {test_accuracy}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=200)
                    
                    if abs(train_accuracy - test_accuracy)>0.1 or (train_accuracy<0.6 and test_accuracy<0.6):
                        Label(window6,text='BAD MODEL',font=('Helvetica',10,'bold'),bg='red',relief='solid').place(x=880,y=250)

                    elif (train_accuracy<test_accuracy) and abs(train_accuracy-test_accuracy)>0.1:
                        Label(window6,text='GOOD MODEL',font=('Helvetica',10,'bold'),bg='green',relief='solid').place(x=880,y=250)

                    elif train_accuracy>0.85 and test_accuracy >0.85:
                        Label(window6,text='VERY GOOD MODEL',font=('Helvetica',10,'bold'),bg='green',relief='solid').place(x=880,y=250)

                    return train_accuracy,test_accuracy,pred
                elif(combo.get()=="Multiple Linear Regression"):
                    x = data.iloc[:, indep].values  
                    y = data.iloc[:, dep].values

                    
                    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
                    regressor=LinearRegression()
                    regressor.fit(x_train,np.ravel(y_train))
                    train_accuracy=regressor.score(x_train,y_train)
                    test_accuracy=regressor.score(x_test,y_test)
                    
                    pred = regressor.predict([result])
                    Label(window6,text=str(pred),font=('Helvetica',10,'bold'),bg='light blue', relief='solid').place(x=880,y=580)
                    Label(window6,text=f'Train accuracy : {train_accuracy}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=150)
                    Label(window6,text=f'Test accuracy : {test_accuracy}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=200)
                    
                    if abs(train_accuracy - test_accuracy)>0.1 or (train_accuracy<0.6 and test_accuracy<0.6):
                        Label(window6,text='BAD MODEL',font=('Helvetica',10,'bold'),bg='red',relief='solid').place(x=880,y=250)

                    elif (train_accuracy<test_accuracy) and abs(train_accuracy-test_accuracy)>0.1:
                        Label(window6,text='GOOD MODEL',font=('Helvetica',10,'bold'),bg='green',relief='solid').place(x=880,y=250)

                    elif train_accuracy>0.85 and test_accuracy >0.85:
                        Label(window6,text='VERY GOOD MODEL',font=('Helvetica',10,'bold'),bg='green',relief='solid').place(x=880,y=250)

                    return train_accuracy,test_accuracy,pred

                elif(combo.get()=="Polynomial Regression"):
                    x = data.iloc[:, indep].values  
                    y = data.iloc[:, dep].values
                    
                    global poly_reg,x_poly

                    poly_reg = PolynomialFeatures(degree=int(entryp.get()))
                    x_poly = poly_reg.fit_transform(x)
                   
                    regressor = LinearRegression()
                    regressor.fit(x_poly,y)              
                    train_accuracy = regressor.score(x_poly,y)
                    
                    pred = regressor.predict(poly_reg.fit_transform([result]))
                    

                    Label(window6,text=str(pred),font=('Helvetica',10,'bold'),bg='light blue', relief='solid').place(x=880,y=580)
                    Label(window6,text=f'Train accuracy : {train_accuracy}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=150)
                    
                    
                elif(combo.get()=="SVM Regression"):
                    
                    x = data.iloc[:, indep].values  
                    y = data.iloc[:, dep].values
                    
                    global sc_x,sc_y
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

                    sc_x = StandardScaler()
                    sc_y = StandardScaler()
                    x_train = sc_x.fit_transform(x_train)
                    x_test = sc_x.fit_transform(x_test)
                    y_train = sc_y.fit_transform(y_train)
                    y_test = sc_y.fit_transform(y_test)

                    regressor = SVR(kernel='rbf')
                    regressor.fit(x_train,y_train)
                    y_pred=regressor.predict(x_test)

                    train_accuracy = regressor.score(x_train,y_train)
                    test_accuracy = regressor.score(x_test,y_test)
                    
                    pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(([result]))))
                    Label(window6,text=str(pred),font=('Helvetica',10,'bold'),bg='light blue', relief='solid').place(x=880,y=580)
                    Label(window6,text=f'Train accuracy : {train_accuracy}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=150)
                    Label(window6,text=f'Test accuracy : {test_accuracy}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=200)
                    
                    if abs(train_accuracy - test_accuracy)>0.1 or (train_accuracy<0.6 and test_accuracy<0.6):
                        Label(window6,text='BAD MODEL',font=('Helvetica',10,'bold'),bg='red',relief='solid').place(x=880,y=250)

                    elif (train_accuracy<test_accuracy) and abs(train_accuracy-test_accuracy)>0.1:
                        Label(window6,text='GOOD MODEL',font=('Helvetica',10,'bold'),bg='green',relief='solid').place(x=880,y=250)

                    elif train_accuracy>0.85 and test_accuracy >0.85:
                        Label(window6,text='VERY GOOD MODEL',font=('Helvetica',10,'bold'),bg='green',relief='solid').place(x=880,y=250)

                    return train_accuracy,test_accuracy,pred
                
                elif(combo.get()=='Decision Tree Regression'):
                    x = data.iloc[:, indep].values  
                    y = data.iloc[:, dep].values
                    
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
                    
                    regressor = DecisionTreeRegressor(criterion='mse',random_state=0)
                    regressor.fit(x_train,y_train)
                    y_pred=regressor.predict(x_test)
                    
                    train_accuracy=regressor.score(x_train,y_train)
                    test_accuracy=regressor.score(x_test,y_test)
                    
                    pred = regressor.predict([result])
                    Label(window6,text=str(pred),font=('Helvetica',10,'bold'),bg='light blue', relief='solid').place(x=880,y=580)
                    Label(window6,text=f'Train accuracy : {train_accuracy}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=150)
                    Label(window6,text=f'Test accuracy : {test_accuracy}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=200)
                    
                    if abs(train_accuracy - test_accuracy)>0.1 or (train_accuracy<0.6 and test_accuracy<0.6):
                        Label(window6,text='BAD MODEL',font=('Helvetica',10,'bold'),bg='red',relief='solid').place(x=880,y=250)

                    elif (train_accuracy<test_accuracy) and abs(train_accuracy-test_accuracy)>0.1:
                        Label(window6,text='GOOD MODEL',font=('Helvetica',10,'bold'),bg='green',relief='solid').place(x=880,y=250)

                    elif train_accuracy>0.85 and test_accuracy >0.85:
                        Label(window6,text='VERY GOOD MODEL',font=('Helvetica',10,'bold'),bg='green',relief='solid').place(x=880,y=250)

                    return train_accuracy,test_accuracy,pred
                
                elif(combo.get()=="Random Forest Regression"):
                    x = data.iloc[:, indep].values  
                    y = data.iloc[:, dep].values
                    
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
                    
                    regressor = RandomForestRegressor(n_estimators=8,random_state=0 ) 
                    regressor.fit(x_train,y_train)
                    y_pred=regressor.predict(x_test)
                    
                    train_accuracy=regressor.score(x_train,y_train)
                    test_accuracy=regressor.score(x_test,y_test)
                    
                    pred = regressor.predict([result])
                    Label(window6,text=str(pred),font=('Helvetica',10,'bold'),bg='light blue', relief='solid').place(x=880,y=580)
                    Label(window6,text=f'Train accuracy : {train_accuracy}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=150)
                    Label(window6,text=f'Test accuracy : {test_accuracy}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=200)
                    
                    if abs(train_accuracy - test_accuracy)>0.1 or (train_accuracy<0.6 and test_accuracy<0.6):
                        Label(window6,text='BAD MODEL',font=('Helvetica',10,'bold'),bg='red',relief='solid').place(x=880,y=250)

                    elif (train_accuracy<test_accuracy) and abs(train_accuracy-test_accuracy)>0.1:
                        Label(window6,text='GOOD MODEL',font=('Helvetica',10,'bold'),bg='green',relief='solid').place(x=880,y=250)

                    elif train_accuracy>0.85 and test_accuracy >0.85:
                        Label(window6,text='VERY GOOD MODEL',font=('Helvetica',10,'bold'),bg='green',relief='solid').place(x=880,y=250)

                    return train_accuracy,test_accuracy,pred
                    
                else:#Decision tree regression and random forest regression
                    # regressor = LinearRegression()
                    # regressor.fit(x_train, y_train)
                    # pred = regressor.predict([result])
                    print('Sorry Something Error Check Your input')
                
####classification    
            elif(algos=="Classification"):
                if (combo2.get() == "Logistic Regression"):
                    
                    x = data.iloc[:, indep].values
                    y = data.iloc[:, dep].values
                    
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

                    global sc
                    sc = StandardScaler()
                    x_train = sc.fit_transform(x_train)
                    x_test = sc.transform(x_test)

                    global classifier
                    classifier = LogisticRegression(random_state=0)
                    classifier.fit(x_train,y_train)

                    y_pred = classifier.predict(x_test)

                    cm = confusion_matrix(y_test,y_pred)
                    global acc
                    acc = (sum(np.diag(cm))/len(y_test))
                    pred = classifier.predict(sc.transform([result]))
                    
                    train_accuracy=accuracy_score(np.ravel(y_train),classifier.predict(x_train))
                    test_accuracy=accuracy_score(np.ravel(y_test),classifier.predict(x_test))

                    
                    Label(window6,text=str(pred),font=('Helvetica',10,'bold'),bg='light blue', relief='solid').place(x=880,y=580)
                    Label(window6,text=f'Train accuracy : {train_accuracy}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=150)
                    Label(window6,text=f'Test accuracy : {test_accuracy}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=200)
                    Label(window6,text=f'Confusion Matrix\nAcc : {acc}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=250)

                    if abs(train_accuracy - test_accuracy)>0.1 or (train_accuracy<0.6 and test_accuracy<0.6):
                        Label(window6,text='BAD MODEL',font=('Helvetica',10,'bold'),bg='red',relief='solid').place(x=880,y=310)

                    elif (train_accuracy<test_accuracy) and abs(train_accuracy-test_accuracy)>0.1:
                        Label(window6,text='GOOD MODEL',font=('Helvetica',10,'bold'),bg='green',relief='solid').place(x=880,y=310)

                    elif train_accuracy>0.85 and test_accuracy >0.85:
                        Label(window6,text='VERY GOOD MODEL',font=('Helvetica',10,'bold'),bg='green',relief='solid').place(x=880,y=310)

                    return train_accuracy,test_accuracy,pred
                
                elif(combo2.get()=="Naive Bayes Classification"):
                    x = data.iloc[:, indep].values
                    y = data.iloc[:, dep].values
                    
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

                    # global sc
                    sc = StandardScaler()
                    x_train = sc.fit_transform(x_train)
                    x_test = sc.transform(x_test)

                    #global classifier
                    classifier = GaussianNB()
                    classifier.fit(x_train,y_train)

                    y_pred = classifier.predict(x_test)

                    cm = confusion_matrix(y_test,y_pred)
                    #global acc
                    acc = (sum(np.diag(cm))/len(y_test))
                    pred = classifier.predict(sc.transform([result]))
                    
                    train_accuracy=accuracy_score(np.ravel(y_train),classifier.predict(x_train))
                    test_accuracy=accuracy_score(np.ravel(y_test),classifier.predict(x_test))

                    
                    Label(window6,text=str(pred),font=('Helvetica',10,'bold'),bg='light blue', relief='solid').place(x=880,y=580)
                    Label(window6,text=f'Train accuracy : {train_accuracy}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=150)
                    Label(window6,text=f'Test accuracy : {test_accuracy}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=200)
                    Label(window6,text=f'Confusion Matrix\nAcc : {acc}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=250)

                    if abs(train_accuracy - test_accuracy)>0.1 or (train_accuracy<0.6 and test_accuracy<0.6):
                        Label(window6,text='BAD MODEL',font=('Helvetica',10,'bold'),bg='red',relief='solid').place(x=880,y=310)

                    elif (train_accuracy<test_accuracy) and abs(train_accuracy-test_accuracy)>0.1:
                        Label(window6,text='GOOD MODEL',font=('Helvetica',10,'bold'),bg='green',relief='solid').place(x=880,y=310)

                    elif train_accuracy>0.85 and test_accuracy >0.85:
                        Label(window6,text='VERY GOOD MODEL',font=('Helvetica',10,'bold'),bg='green',relief='solid').place(x=880,y=310)

                    return train_accuracy,test_accuracy,pred
                    
                elif(combo2.get()=="K-Nearest Neighbour Classification"):
                    x = data.iloc[:, indep].values
                    y = data.iloc[:, dep].values
                    
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

                    #global sc
                    sc = StandardScaler()
                    x_train = sc.fit_transform(x_train)
                    x_test = sc.transform(x_test)

                    #global classifier
                    classifier = KNeighborsClassifier(n_neighbors=10,metric='minkowski',p=2)
                    classifier.fit(x_train,y_train)

                    y_pred = classifier.predict(x_test)

                    cm = confusion_matrix(y_test,y_pred)
                    #global acc
                    acc = (sum(np.diag(cm))/len(y_test))
                    pred = classifier.predict(sc.transform([result]))
                    
                    train_accuracy=accuracy_score(np.ravel(y_train),classifier.predict(x_train))
                    test_accuracy=accuracy_score(np.ravel(y_test),classifier.predict(x_test))

                    
                    Label(window6,text=str(pred),font=('Helvetica',10,'bold'),bg='light blue', relief='solid').place(x=880,y=580)
                    Label(window6,text=f'Train accuracy : {train_accuracy}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=150)
                    Label(window6,text=f'Test accuracy : {test_accuracy}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=200)
                    Label(window6,text=f'Confusion Matrix\nAcc : {acc}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=250)

                    if abs(train_accuracy - test_accuracy)>0.1 or (train_accuracy<0.6 and test_accuracy<0.6):
                        Label(window6,text='BAD MODEL',font=('Helvetica',10,'bold'),bg='red',relief='solid').place(x=880,y=310)

                    elif (train_accuracy<test_accuracy) and abs(train_accuracy-test_accuracy)>0.1:
                        Label(window6,text='GOOD MODEL',font=('Helvetica',10,'bold'),bg='green',relief='solid').place(x=880,y=310)

                    elif train_accuracy>0.85 and test_accuracy >0.85:
                        Label(window6,text='VERY GOOD MODEL',font=('Helvetica',10,'bold'),bg='green',relief='solid').place(x=880,y=310)

                    return train_accuracy,test_accuracy,pred

                    
                     
                elif(combo2.get()=="SVM Classification"):
                    x = data.iloc[:, indep].values
                    y = data.iloc[:, dep].values
                    
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

                    #global sc
                    sc = StandardScaler()
                    x_train = sc.fit_transform(x_train)
                    x_test = sc.transform(x_test)

                    #global classifier
                    classifier = SVC(C=100,kernel='rbf',random_state=0,gamma=0.5)
                    classifier.fit(x_train,y_train)

                    y_pred = classifier.predict(x_test)

                    cm = confusion_matrix(y_test,y_pred)
                    #global acc
                    acc = (sum(np.diag(cm))/len(y_test))
                    pred = classifier.predict(sc.transform([result]))
                    
                    train_accuracy=accuracy_score(np.ravel(y_train),classifier.predict(x_train))
                    test_accuracy=accuracy_score(np.ravel(y_test),classifier.predict(x_test))

                    
                    Label(window6,text=str(pred),font=('Helvetica',10,'bold'),bg='light blue', relief='solid').place(x=880,y=580)
                    Label(window6,text=f'Train accuracy : {train_accuracy}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=150)
                    Label(window6,text=f'Test accuracy : {test_accuracy}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=200)
                    Label(window6,text=f'Confusion Matrix\nAcc : {acc}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=250)

                    if abs(train_accuracy - test_accuracy)>0.1 or (train_accuracy<0.6 and test_accuracy<0.6):
                        Label(window6,text='BAD MODEL',font=('Helvetica',10,'bold'),bg='red',relief='solid').place(x=880,y=310)

                    elif (train_accuracy<test_accuracy) and abs(train_accuracy-test_accuracy)>0.1:
                        Label(window6,text='GOOD MODEL',font=('Helvetica',10,'bold'),bg='green',relief='solid').place(x=880,y=310)

                    elif train_accuracy>0.85 and test_accuracy >0.85:
                        Label(window6,text='VERY GOOD MODEL',font=('Helvetica',10,'bold'),bg='green',relief='solid').place(x=880,y=310)

                    return train_accuracy,test_accuracy,pred
                elif(combo2.get()=="Decision Tree Classification"):
                    x = data.iloc[:, indep].values
                    y = data.iloc[:, dep].values
                    
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

                    #global sc
                    sc = StandardScaler()
                    x_train = sc.fit_transform(x_train)
                    x_test = sc.transform(x_test)

                    #global classifier
                    classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
                    classifier.fit(x_train,y_train)

                    y_pred = classifier.predict(x_test)

                    cm = confusion_matrix(y_test,y_pred)
                    #global acc
                    acc = (sum(np.diag(cm))/len(y_test))
                    pred = classifier.predict(sc.transform([result]))
                    
                    train_accuracy=accuracy_score(np.ravel(y_train),classifier.predict(x_train))
                    test_accuracy=accuracy_score(np.ravel(y_test),classifier.predict(x_test))

                    
                    Label(window6,text=str(pred),font=('Helvetica',10,'bold'),bg='light blue', relief='solid').place(x=880,y=580)
                    Label(window6,text=f'Train accuracy : {train_accuracy}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=150)
                    Label(window6,text=f'Test accuracy : {test_accuracy}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=200)
                    Label(window6,text=f'Confusion Matrix\nAcc : {acc}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=250)

                    if abs(train_accuracy - test_accuracy)>0.1 or (train_accuracy<0.6 and test_accuracy<0.6):
                        Label(window6,text='BAD MODEL',font=('Helvetica',10,'bold'),bg='red',relief='solid').place(x=880,y=310)

                    elif (train_accuracy<test_accuracy) and abs(train_accuracy-test_accuracy)>0.1:
                        Label(window6,text='GOOD MODEL',font=('Helvetica',10,'bold'),bg='green',relief='solid').place(x=880,y=310)

                    elif train_accuracy>0.85 and test_accuracy >0.85:
                        Label(window6,text='VERY GOOD MODEL',font=('Helvetica',10,'bold'),bg='green',relief='solid').place(x=880,y=310)

                    return train_accuracy,test_accuracy,pred
                elif(combo2.get()=="Random Forest Classification"):
                    x = data.iloc[:, indep].values
                    y = data.iloc[:, dep].values
                    
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

                    #global sc
                    sc = StandardScaler()
                    x_train = sc.fit_transform(x_train)
                    x_test = sc.transform(x_test)

                    #global classifier
                    classifier = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)
                    classifier.fit(x_train,y_train)

                    y_pred = classifier.predict(x_test)

                    cm = confusion_matrix(y_test,y_pred)
                    #global acc
                    acc = (sum(np.diag(cm))/len(y_test))
                    pred = classifier.predict(sc.transform([result]))
                    
                    train_accuracy=accuracy_score(np.ravel(y_train),classifier.predict(x_train))
                    test_accuracy=accuracy_score(np.ravel(y_test),classifier.predict(x_test))

                    
                    Label(window6,text=str(pred),font=('Helvetica',10,'bold'),bg='light blue', relief='solid').place(x=880,y=580)
                    Label(window6,text=f'Train accuracy : {train_accuracy}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=150)
                    Label(window6,text=f'Test accuracy : {test_accuracy}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=200)
                    Label(window6,text=f'Confusion Matrix\nAcc : {acc}',font=('Helvetica',10,'bold'),bg='light blue',relief='solid').place(x=880,y=250)

                    if abs(train_accuracy - test_accuracy)>0.1 or (train_accuracy<0.6 and test_accuracy<0.6):
                        Label(window6,text='BAD MODEL',font=('Helvetica',10,'bold'),bg='red',relief='solid').place(x=880,y=310)

                    elif (train_accuracy<test_accuracy) and abs(train_accuracy-test_accuracy)>0.1:
                        Label(window6,text='GOOD MODEL',font=('Helvetica',10,'bold'),bg='green',relief='solid').place(x=880,y=310)

                    elif train_accuracy>0.85 and test_accuracy >0.85:
                        Label(window6,text='VERY GOOD MODEL',font=('Helvetica',10,'bold'),bg='green',relief='solid').place(x=880,y=310)

                    return train_accuracy,test_accuracy,pred
                     
                
                   
                   
            else:
                    print('Sorry something wrong input check it')
               
    
    
        button61 = Button(window6, text="Predict", command=predict)
        button61.place(x=300,y=z+50,height=40,width=100)
        
        
     
        label8=tkinter.Label(window6,relief='solid')
        label8.pack()
        label8.place(x=860,y=50,height=100,width=400)
        label8.configure(text='------Summary-----',font=('helvetica',20,'bold'))
        
        
        label10=tkinter.Label(window6,relief='solid')
        label10.pack()
        label10.place(x=860,y=130,height=500,width=400)
        
        label11=tkinter.Label(window6,relief='solid')
        label11.pack()
        label11.place(x=860,y=500,width=400,height=50)
        label11.configure(text='New observation result',font=('helvetica',12,'bold'))
    
        window6.mainloop()

    def window7(self):
        window7 = Tk()
        window7.configure(bg="lightblue")
        window7.state('zoomed')
        window7.title("ML-Algorithms")
        
        
        label = Label(window7,text='Clustering Algorithms',bg='lightblue',font=('helvetica',16,'bold'))
        label.place(x=670,y=330)
        
        label2 = Label(window7,relief='solid').place(x=500,y=300,height=200,width=400)
        
        label3 = Label(window7,text='Click here to get Cluster plot and result',font=('helvetica',14,'bold'))
        label3.place(x=520,y=350,height=40)
        
        
        label1 = Label(window7, text="Index Name",  bg="lightblue", font=("helvetica", 20, 'bold'))
        label1.place(x=70, y=100, height=30)
    
        z=140
        global entry1, entries1
        entries1 = []
        for i in indep1:
            label2 = Label(window7, text=i,bg="lightblue", font=("helvetica", 15, 'bold'))
            label2.place(x=75, y=z)
    
            entry1 = Entry(window7)
            entry1.place(x=290, y=z, height=30, width=150)
            entries1.append(entry1)
            z = z + 25
    
        label3 = Label(window7, text="Index Values", bg="lightblue", font=("helvetica", 20, 'bold'))
        label3.place(x=295, y=90)
        
        def cluster():
            if(algos=="clustering"):
                global resultc
                resultc=[]
                for entry in entries1:
                    resultc.append(entry.get())
                
                if(combo3.get()=="KMeans Clustering"):
                    
                    x_normal = data.iloc[:, indep].values
                    
                    from sklearn.decomposition import KernelPCA
                    kpca=KernelPCA(n_components=2)
                    
                    x=kpca.fit_transform(x_normal)
                    
                    kmeans=KMeans(n_clusters=int(clusterkmeans.get()) ,random_state=0)
                    y_kmeans=kmeans.fit_predict(x)
                    y_predict=kmeans.predict(kpca.transform([resultc]))
                    
                    label4 = Label(window7,text=f'New obs result:{y_predict}',bg='lightgreen',relief='solid')
                    label4.place(x=245,y=z+35,height=30)
                    
                    
                    def plotkmeans():
                     
                        for i in range(0,int(clusterkmeans.get())):
                            if i<=0:
                 
                                plt.figure(dpi=300)
                                plt.scatter(x[:,0],x[:,1],c=kmeans.labels_, cmap='rainbow')
                                plt.scatter(x[y_kmeans==i,0],x[y_kmeans==i,1],s=10)
    
                                plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=200,
                                            c='yellow')
                                plt.title('clusters')
                                plt.xlabel('x')
                                plt.ylabel('y')
                                plt.legend()
                                plt.show()
                    button=Button(window7,text='Cluster',command=plotkmeans())
                    
                    
                
                elif(combo3.get()=="Hierarchical Clustering"):
                    
                    
                    x_normal = data.iloc[:, indep].values
                    from sklearn.decomposition import KernelPCA
                    kpca=KernelPCA(n_components=2)
                    
                    x=kpca.fit_transform(x_normal)
                    
                                    
                    import scipy.cluster.hierarchy as sch
                    dentogram=sch.dendrogram(sch.linkage(x,method='centroid'))
                    
                    
                    from sklearn.cluster import AgglomerativeClustering
                    hc=AgglomerativeClustering(n_clusters=int(clustersch.get()))
                    y=hc.fit_predict(x)
                    
                    
                   
                    np.unique(y)
                    def plotsch():
                        import random
                      
                        for i in range(0,int(clustersch.get())):
                            if i<=0:
                        
                                plt.figure(dpi=300)
                                plt.scatter(x[:,0],x[:,1],s=15,c=hc.labels_, cmap='rainbow')
                                plt.scatter(x[y==i,0],x[y==i,1],s=10)
                                
                                plt.xlabel('x')
                                plt.ylabel('y')
                                plt.title('clusters')
                                plt.legend()
                                plt.show()
                        
                    button=Button(window7,text='Cluster',command=plotsch())
                    
            else:
                print('something wrong input check it')
            
        button=Button(window7,text='Predict',bg='lightblue',command=cluster)
        button.place(x=620,y=400,width=150,height=60)

        window7.mainloop()
        
w=window()
w.window1()   