### Setup

    library(AppliedPredictiveModeling);library(caret);library(rpart);library(pgmm);library(ElemStatLearn)
    library(rpart.plot);library(rattle);library(e1071);library(forecast);library(gbm);library(plyr)
    library(lubridate);library(dplyr);library(ggplot2);library(lattice);library(randomForest)

OVERVIEW
========

#### Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).

DOWNLOAD DATA (From working directory)
--------------------------------------

##### The training data for this project are available here: <https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

##### The test data are available here: <https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

            traindata<-read.csv("pml-training.csv")
            testdata<-read.csv("pml-testing.csv")
            traina<-tbl_df(traindata)
            testa<-tbl_df(testdata)
            dim(traina)

    ## [1] 19622   160

            dim(testa)

    ## [1]  20 160

how many different types of classe exist
========================================

            trainb<-group_by(traina,classe)
            trainc<-arrange(trainb,classe)
            traind<-summarise(trainc,n_distinct(classe))
            traind[,1]

    ## # A tibble: 5 × 1
    ##   classe
    ##   <fctr>
    ## 1      A
    ## 2      B
    ## 3      C
    ## 4      D
    ## 5      E

### Remove near zero variables and Remove variable with mostly (95%+) NA's. Also remove first five columns(id only)

    nza<-nearZeroVar(traina)
    traine<-traina[,-nza]
    testb<-testa[,-nza]
    mostNA<-sapply(traine, function(x) mean(is.na(x)))>=.95
    trainf<-traine[,mostNA==F]
    testc<-testb[,mostNA==F]
    Train<-trainf[,-(1:5)]
    Test<-testc[,-(1:5)]

    dim(Train)

    ## [1] 19622    54

    dim(Test)

    ## [1] 20 54

### How many observations of each classe are found?

            NumA<-nrow( filter(Train,classe=="A"))
            NumA

    ## [1] 5580

            NumB<-nrow( filter(Train,classe=="B"))
            NumB

    ## [1] 3797

            NumC<-nrow( filter(Train,classe=="C"))
            NumC

    ## [1] 3422

            NumD<-nrow( filter(Train,classe=="D"))
            NumD

    ## [1] 3216

            NumE<-nrow( filter(Train,classe=="E"))
            NumE

    ## [1] 3607

            Totalclasse<-NumA+NumB+NumC+NumD+NumE
            Totalclasse

    ## [1] 19622

### Split main training database into two training datasets

    set.seed(100)
    split<-createDataPartition(y=Train$classe,p=.7,list = F)
    trainbig<-Train[split,]
    trainsmall<-Train[-split,]

    dim(trainbig)

    ## [1] 13737    54

    dim(trainsmall)

    ## [1] 5885   54

Use machine learning functions to creat a model
-----------------------------------------------

### Try Model Building with "Random Forest"

    set.seed(200)
    control<-trainControl(method = "cv",number = 3,verboseIter = FALSE)

    RFmodelbig<-train(classe~.,data=trainbig,method="rf",trControl=control)
    RFmodelbig$finalModel

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = param$mtry) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 27
    ## 
    ##         OOB estimate of  error rate: 0.24%
    ## Confusion matrix:
    ##      A    B    C    D    E  class.error
    ## A 3904    1    0    0    1 0.0005120328
    ## B    4 2650    2    2    0 0.0030097818
    ## C    0    6 2390    0    0 0.0025041736
    ## D    0    0    8 2243    1 0.0039964476
    ## E    0    1    0    7 2517 0.0031683168

### Assess accuracy of RF model

    RFpred<-predict(RFmodelbig,newdata = trainsmall)
    RF<-confusionMatrix(trainsmall$classe,RFpred)
    RF

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1674    0    0    0    0
    ##          B    6 1133    0    0    0
    ##          C    0    0 1026    0    0
    ##          D    0    0    7  957    0
    ##          E    0    0    0    0 1082
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9978          
    ##                  95% CI : (0.9962, 0.9988)
    ##     No Information Rate : 0.2855          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9972          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9964   1.0000   0.9932   1.0000   1.0000
    ## Specificity            1.0000   0.9987   1.0000   0.9986   1.0000
    ## Pos Pred Value         1.0000   0.9947   1.0000   0.9927   1.0000
    ## Neg Pred Value         0.9986   1.0000   0.9986   1.0000   1.0000
    ## Prevalence             0.2855   0.1925   0.1755   0.1626   0.1839
    ## Detection Rate         0.2845   0.1925   0.1743   0.1626   0.1839
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
    ## Balanced Accuracy      0.9982   0.9994   0.9966   0.9993   1.0000

### Try Model Building with "Generalized Boosted Model"(GBM) and assess accuracy GBM model

    set.seed(200)
    GBMmodelbig<-train(classe~.,data=trainbig,method="gbm",trControl=control,verbose=FALSE)
    GBMpred<-predict(GBMmodelbig,newdata = trainsmall)
    GBM<-confusionMatrix(trainsmall$classe,GBMpred)
    GBMmodelbig$finalModel

    ## A gradient boosted model with multinomial loss function.
    ## 150 iterations were performed.
    ## There were 53 predictors of which 43 had non-zero influence.

    GBM

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1674    0    0    0    0
    ##          B   14 1108   17    0    0
    ##          C    0   11 1007    8    0
    ##          D    0    3   16  943    2
    ##          E    0    6    7    7 1062
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.9845         
    ##                  95% CI : (0.981, 0.9875)
    ##     No Information Rate : 0.2868         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.9804         
    ##  Mcnemar's Test P-Value : NA             
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9917   0.9823   0.9618   0.9843   0.9981
    ## Specificity            1.0000   0.9935   0.9961   0.9957   0.9959
    ## Pos Pred Value         1.0000   0.9728   0.9815   0.9782   0.9815
    ## Neg Pred Value         0.9967   0.9958   0.9918   0.9970   0.9996
    ## Prevalence             0.2868   0.1917   0.1779   0.1628   0.1808
    ## Detection Rate         0.2845   0.1883   0.1711   0.1602   0.1805
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
    ## Balanced Accuracy      0.9959   0.9879   0.9789   0.9900   0.9970

### Both RF and GBM had good accuracy - RF was the better of the two methods and therefore used RF (99.8% accurate) to predict the classe for each of the twenty observations in the "test" database.

    FinalPred<-predict(RFmodelbig,newdata = Test)
    FinalPreddf<-data.frame(Test_ID=Test$problem_id,PredictClass=FinalPred)
    FinalPreddf

    ##    Test_ID PredictClass
    ## 1        1            B
    ## 2        2            A
    ## 3        3            B
    ## 4        4            A
    ## 5        5            A
    ## 6        6            E
    ## 7        7            D
    ## 8        8            B
    ## 9        9            A
    ## 10      10            A
    ## 11      11            B
    ## 12      12            C
    ## 13      13            B
    ## 14      14            A
    ## 15      15            E
    ## 16      16            E
    ## 17      17            A
    ## 18      18            B
    ## 19      19            B
    ## 20      20            B

Conclusion - the Random Forest Method of model building proved itself very accurate (over 99%) for this project. Combining multiple models was not considered essential since the RF method proved itself so accurate.
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
