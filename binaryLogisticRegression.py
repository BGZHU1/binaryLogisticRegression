import numpy as np
import pylab
from sklearn import datasets
from scipy import optimize, special
from scipy.optimize import check_grad

#gradientList=[]
#calculate individual probability
#learning_rate=0.001

def sigmoid(x,w):

#five parameters - first one is one

#get the function represenation for each datapoint(w1*x1+.....w4*x4+w0*x0)
#w0*x0 is the interception

    wTx=np.sum((w.T*x),axis=1)
    #print(w)
    #print(x)

    #print("wTx",wTx)

#calculate different probability for each data point (two classes)
    p0=1/(1+np.exp(-wTx))
    p1=np.exp(-wTx)/(1+np.exp(-wTx))

    #print('p0',p0)
    #print('p1',p1)


    return p0,p1






def crossEntropy(prob,target):
    # with the label function
    cross_entropy= (target*np.log(prob[0])+(1-target)*np.log(1-prob[0]))/len(prob)
    #print("part1",target*np.log(prob[0]))
    cross_entropy=-np.sum(cross_entropy)
    #print('cross_entropy',cross_entropy)


    return cross_entropy



#change later
def crossEntropy_Regularization(prob,target,x,w):
    # with the label function
    cross_entropy= (target*np.log(prob[0])+(1-target)*np.log(1-prob[0]))/len(prob)
    #print("part1",target*np.log(prob[0]))
    cross_entropy=-np.sum(cross_entropy)+0.5*(w.T*w)
    #print('cross_entropy',cross_entropy)

    return cross_entropy




def gradient_calculation(prob,target,x,param):
    gradient=0

    gradient=np.sum((target-prob)*x.T)

    return gradient

def updateParam(target,prob,param):
    n=-np.sum((target-prob)*x.T,axis=1)
    param=param-learning_rate*n
    return param




def crossEntropy2(*args):
    # with the label function
    prob=args[0]
    target=args[1]
    cross_entropy= (target*np.log(prob)+(1-target)*np.log(1-prob))/len(prob)
    #print("part1",target*np.log(prob[0]))
    cross_entropy=-np.sum(cross_entropy)
    #print('cross_entropy',cross_entropy)


    return cross_entropy


def gradient_calculation2(*args):
    gradient=0
    prob=args[0]
    target=args[1]
    x=args[2]

    gradient0=np.sum((target-prob)*x.T)
    gradient=np.array([gradient0])



    return gradient



def crossEntropy3(x,a,b,c):
        # with the label function
    prob=a
    target=b
    cross_entropy= (target*np.log(prob)+(1-target)*np.log(1-prob))/len(prob)
        #print("part1",target*np.log(prob[0]))
    cross_entropy=-np.sum(cross_entropy)
        #print('cross_entropy',cross_entropy)


    return cross_entropy


def gradient_calculation3(x,a,b,c):
    gradient=0
    prob=a
    target=b
    input=c

    gradient0=np.sum((target-prob)*input.T)
    gradient=np.array([gradient0])

        #gradientList=[]
        #for i in range(0,len(x)):
            #n=-(target-prob)[i]*x[i]
            #update weights from param
            #gradientList.append(n)
            #param[i]=param[i]-learning_rate*n


            #g=np.sum(n)

            #gradient+=g


        #print('gradient',gradient)
        #print('gradientlist',gradientList)


    return gradient


if __name__ == "__main__":
    iris=datasets.load_iris()
     #print(iris.target)
     #print(iris.data)
    x=iris.data


    #create a column to add to x
    new_col=np.reshape(np.ones((1,len(x))),(len(x),1))
    #print(new_col)

    #add column into the x
    input=np.concatenate((x,new_col),1)

    #print('x',x)

    #reshape(input,numberOfElement)
    param=np.ones(len(input[0]))/10
    #param=np.reshape(param,len(x))

    #for class 1 and class 2,3
    #need more implementation here for class 2,1,3 and class 3, 1,2
    target1=iris.target
        #create label array

    targets=np.array([1 if x==0 else 0 for x in target1])
    #print(target)
    #print('labelList',labelList)

    #for cross validation - 2 folds
    for i in range (0,2) :


        #training
        input1=input[0:75]
        input2=input[75:149]
        input_group=[input1,input2]


        t1=targets[0:75]
        t2=targets[75:149]
        target_group=[t1,t2]

        x=input_group[i]
        target=target_group[i]

        #validationData
        validation_group=[input2,input1]
        validation_target=[t2,t1]
        validationData=validation_group[i]



       #newton method
        #chekc the gradient

         #process of quasi-newton fmin_l_bfgs_b
        #m = optimize.fmin_l_bfgs_b(crossEntropy, x0 =0, fprime=gradient_calculation, args=(prob,target,prob,target,x,param), bounds = None)



       #process of gradient descent

        prob=sigmoid(x, param)
        #print('prob',prob)
        #print('prob',prob[0])
        a=np.array(prob[0])
        b=np.array(target)
        c=np.array(x)
        check_grad=optimize.check_grad(crossEntropy3,gradient_calculation3,[-10,10],a,b,c)
        print("check_grad",check_grad)

        inital_array=np.array([100])
        fmin_l_bfs_b = optimize.fmin_l_bfgs_b(crossEntropy2, x0 = inital_array, fprime=gradient_calculation2, args=(np.array(prob[0]),np.array(target),np.array(x)), bounds = None)
        print('fmin_l_bfs_b',fmin_l_bfs_b)

        gradient=gradient_calculation(prob[0],target,x,param)
        crossentropy = crossEntropy(prob[0],target)




        #gradient_old=gradient_calculation(w,result[0],result[1],target,x)
        learning_rate=0.001

        old_gradient=gradient

            #update the param
        param=updateParam(target,prob[0],param)

        prob=sigmoid(x, param)

        #recaculate gradient

        new_gradient=gradient_calculation(prob[0],target,x,param)

        counts=[]
        crossE=[]
        crossEntropy_withRegularizaion=[]
        count=0
        #print('different',new_gradient-old_gradient)
        while new_gradient-old_gradient>0.00001:



            count=count+1
            #print(count)
            counts.append(count)
            ce=crossEntropy(np.array(prob),np.array(target))
            cr=crossEntropy_Regularization(np.array(prob),np.array(target),x,param)
            crossE.append(ce)
            crossEntropy_withRegularizaion.append(cr)

            #calculate stepsize


            #step_size=np.reshape(np.array(gradientList),(len(param),len(param[0])))

            #update the param

            #update the param
            param=updateParam(target,prob[0],param)

            prob=sigmoid(x, param)

            #recaculate prob
            prob=sigmoid(x, param)

            #print('gparam',param)
            #print('gprob',prob)

            #recaculate gradient
            old_gradient=new_gradient
            new_gradient=gradient_calculation(prob[0],target,x,param)
            #print('param',param)
            diff=new_gradient-old_gradient
            #print('diif', diff)

        print('param',param)


        result= sigmoid(validationData,param)
        #print('result',result)

        finalResult=result[0]-result[1]
        trueTarget=validation_target[i]
        print('finalresult',finalResult)
        positive=0
        for i in range(0,len(finalResult)):
            if finalResult[i]>0 :
                positive=positive+1
        print('positive',positive)

        true=0
        for i in range(0,len(trueTarget)):
            if trueTarget[i]==1 :
                true=true+1
        print('trueTarget',true)

        print('trueTarget',trueTarget)

        pylab.plot(counts,crossE)
        #pylab.plot(counts,crossEntropy_withRegularizaion)

        pylab.show()


        #cross_validation
