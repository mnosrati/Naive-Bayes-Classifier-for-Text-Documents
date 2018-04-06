tic
% Loading the dataset

[map test_data test_label train_data train_label vocabulary]=LoadDataset;
clc
disp('Dataset is loaded now...');disp(' ');



% 2.1 Learn Naive Bayes Model
%==========================================================================

% 1. P(omega) = # documents in that class / # all documents  --> using train DS
[numTrainDocs tmp]=size(train_label);
[numTestDocs tmp]=size(test_label);
[numClasses,tmp]=hist(train_label,unique(train_label));
numClasses=numClasses';
P=numClasses/numTrainDocs;

for i=1:length(P)
    disp(['P(Omega=' num2str(i) ') = ' num2str(P(i))])
end


% 2. Calculate n: total number of words in all documents in class ?j
n=zeros(length(numClasses),1);
for i=1:length(train_data)
    docNum=train_data(i,1);
    docClass=train_label(docNum,1);
    n(docClass)=n(docClass)+train_data(i,3);
end


% 3. Calculate nk: number of times word wk occurs in all documents in class ?j.
nk=zeros(length(numClasses),length(vocabulary));
for i=1:length(train_data)
    docNum=train_data(i,1);
    docClass=train_label(docNum,1);
    wordIdx=train_data(i,2);
    nk(docClass,wordIdx)=nk(docClass,wordIdx)+train_data(i,3);
end


% 4. Calcualte Maximum Likelihood estimator PMLE(wk|?j) = nk/n 
     %  and Bayesian estimator PBE(wk|?j) = nk+1/n+|V ocabulary| 
PMLE=zeros(length(numClasses),length(vocabulary));
for i=1:length(numClasses) 
    PMLE(i,:)=nk(i,:)/n(i);
end

PBE=zeros(length(numClasses),length(vocabulary));
for i=1:length(numClasses) 
    PBE(i,:)=(nk(i,:)+1)/(n(i)+length(vocabulary));
end

disp(' '); disp('PMLE and PBE are calculated now.'); disp(' ');




% 2.2 Evaluate the Performance of the Classifier
%==========================================================================

%Case 1 --> PMLE on train_data
Pxw1=zeros(length(numClasses),numTrainDocs);
for i=1:length(train_data)
    docIdx=train_data(i,1);
    wordIdx=train_data(i,2);
    wordCount=train_data(i,3);
    Pxw1(:,docIdx)=Pxw1(:,docIdx)+ log(PMLE(:,wordIdx))*wordCount;
end
WNB1=Pxw1;
for i=1:length(numClasses)
    WNB1(i,:)=WNB1(i,:)+log(P(i));
end
disp('WNB1 calculated using PMLE on train_data');


%Case 2 --> PBE on train_data
Pxw2=zeros(length(numClasses),numTrainDocs);
for i=1:length(train_data)
    docIdx=train_data(i,1);
    wordIdx=train_data(i,2);
    wordCount=train_data(i,3);
    Pxw2(:,docIdx)=Pxw2(:,docIdx)+ log(PBE(:,wordIdx))*wordCount;
end
WNB2=Pxw2;
for i=1:length(numClasses)
    WNB2(i,:)=WNB2(i,:)+log(P(i));
end
disp('WNB2 calculated using PBE on train_data');


%Case 3 --> PMLE on test_data
Pxw3=zeros(length(numClasses),numTestDocs);
for i=1:length(test_data)
    docIdx=test_data(i,1);
    wordIdx=test_data(i,2);
    wordCount=test_data(i,3);
    Pxw3(:,docIdx)=Pxw3(:,docIdx)+ log(PMLE(:,wordIdx))*wordCount;
end
WNB3=Pxw3;
for i=1:length(numClasses)
    WNB3(i,:)=WNB3(i,:)+log(P(i));
end
disp('WNB3 calculated using PMLE on test_data');


%Case 4 --> PBE on train_data
Pxw4=zeros(length(numClasses),numTestDocs);
for i=1:length(test_data)
    docIdx=test_data(i,1);
    wordIdx=test_data(i,2);
    wordCount=test_data(i,3);
    Pxw4(:,docIdx)=Pxw4(:,docIdx)+ log(PBE(:,wordIdx))*wordCount;
end
WNB4=Pxw4;
for i=1:length(numClasses)
    WNB4(i,:)=WNB4(i,:)+log(P(i));
end
disp('WNB4 calculated using PBE on test_data');disp(' ');




% 2.1.1 Performance on Training Data
%==========================================================================
% 1. Overall accuracy
[tmp,answerClass1]=max(WNB2);
answerClass1=answerClass1';
accSum=0;
inaccSum=0;
for i=1:length(train_label)
    if(answerClass1(i)-train_label(i)==0)
        accSum=accSum+1;
    else
        inaccSum=inaccSum+1;
    end
end
overallAccuract1=accSum/length(train_label);
disp('For the train_data and PBE:');
disp('---------------------------');
disp(['Overall accuracy = ' num2str(overallAccuract1)]);disp(' ');


% 2. Class accuracy
classAccuracy1=zeros(length(numClasses),1);
totalDocs=zeros(length(numClasses),1);
for i=1:length(train_data)
    docIdx=train_data(i,1);
    totalDocs(train_label(docIdx))=totalDocs(train_label(docIdx))+1;
    if (train_label(docIdx)==answerClass1(docIdx))
        classAccuracy1(train_label(docIdx))=classAccuracy1(train_label(docIdx))+1;
    end
end
classAccuracy1=classAccuracy1./totalDocs;
disp('Class accuracy:');
for i=1:length(numClasses)
    disp(['Group ' num2str(i) ': ' num2str(classAccuracy1(i))]);
end
disp(' ');


% 3. confusion matrix
confusionMatrix1=zeros(length(numClasses));
for i=1:length(train_label)
    confusionMatrix1(train_label(i),answerClass1(i))=confusionMatrix1(train_label(i),answerClass1(i))+1;
end
disp('Confusion matrix: (Vertical: actual classes, horizontal: classification answers)');
confusionMatrix1



% 2.2.2 Performance on Training Data
%==========================================================================
% 1. RBE on test_data

% 1.1 Overall accuracy
[tmp,answerClass2]=max(WNB4);
answerClass2=answerClass2';
accSum=0;
inaccSum=0;
for i=1:length(test_label)
    if(answerClass2(i)-test_label(i)==0)
        accSum=accSum+1;
    else
        inaccSum=inaccSum+1;
    end
end
overallAccuracy2=accSum/length(test_label);
disp('For the test_data and PBE:');
disp('---------------------------');
disp(['Overall accuracy = ' num2str(overallAccuracy2)]);disp(' ');


% 1.2 Class accuracy
classAccuracy2=zeros(length(numClasses),1);
totalDocs=zeros(length(numClasses),1);
for i=1:length(test_data)
    docIdx=test_data(i,1);
    totalDocs(test_label(docIdx))=totalDocs(test_label(docIdx))+1;
    if (test_label(docIdx)==answerClass2(docIdx))
        classAccuracy2(test_label(docIdx))=classAccuracy2(test_label(docIdx))+1;
    end
end
classAccuracy2=classAccuracy2./totalDocs;
disp('Class accuracy:');
for i=1:length(numClasses)
    disp(['Group ' num2str(i) ': ' num2str(classAccuracy2(i))]);
end
disp(' ');


% 1.3 confusion matrix
confusionMatrix2=zeros(length(numClasses));
for i=1:length(test_label)
    confusionMatrix2(test_label(i),answerClass2(i))=confusionMatrix2(test_label(i),answerClass2(i))+1;
end
disp('Confusion matrix: (Vertical: actual classes, horizontal: classification answers)');
confusionMatrix2



% 2. RMLE on test_data

% 2.1 Overall accuracy
[tmp,answerClass3]=max(WNB3);
answerClass3=answerClass3';
accSum=0;
inaccSum=0;
for i=1:length(test_label)
    if(answerClass3(i)-test_label(i)==0)
        accSum=accSum+1;
    else
        inaccSum=inaccSum+1;
    end
end
overallAccuracy3=accSum/length(test_label);
disp('For the test_data and PMLE:');
disp('---------------------------');
disp(['Overall accuracy = ' num2str(overallAccuracy3)]);disp(' ');


% 2.2 Class accuracy
classAccuracy3=zeros(length(numClasses),1);
totalDocs=zeros(length(numClasses),1);
for i=1:length(test_data)
    docIdx=test_data(i,1);
    totalDocs(test_label(docIdx))=totalDocs(test_label(docIdx))+1;
    if (test_label(docIdx)==answerClass3(docIdx))
        classAccuracy3(test_label(docIdx))=classAccuracy3(test_label(docIdx))+1;
    end
end
classAccuracy3=classAccuracy3./totalDocs;
disp('Class accuracy:');
for i=1:length(numClasses)
    disp(['Group ' num2str(i) ': ' num2str(classAccuracy3(i))]);
end
disp(' ');


% 3.3 confusion matrix
confusionMatrix3=zeros(length(numClasses));
for i=1:length(test_label)
    confusionMatrix3(test_label(i),answerClass3(i))=confusionMatrix3(test_label(i),answerClass3(i))+1;
end
disp('Confusion matrix: (Vertical: actual classes, horizontal: classification answers)');
confusionMatrix3


toc

