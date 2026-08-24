%%%%%Student–Learning-Based Optimizer
%%%%%author:Qixiangbo and MaXu
%%Student–Learning-Based Optimizer: An optimization method for continuous non-linear large scale problems
function [GlobalMins,GlobalSolution] = con_SLBO(ProblemStruct)
%%innitialize the problem
if nargin<1
else
    runtime=ProblemStruct.Run;
    maxEvaluation=ProblemStruct.Maxeva;
    n=ProblemStruct.PopSize;
    Dim=ProblemStruct.Dim;
    objfun=ProblemStruct.FunctionName;
    fobj=ProblemStruct.fobj;
    func_num=ProblemStruct.Func_num;
    ub=ProblemStruct.Ud;
    lb=ProblemStruct.Ld;
    aa=0.2;
    bb=0.7;

end
%%%%%%%%%%%%%%%%%%
globalW=1;
historyW=1;
SemesterLength=50;
% maxSemester=30;
% minSemester=5;
errorVal=0.01;
if length(lb)==1
    Ld=ones(1,Dim)*lb;
    Ud=ones(1,Dim)*ub;
else
    Ld=lb;
    Ud=ub;
end
% range=Ud-Ld;
GlobalMins=ones(runtime,maxEvaluation);
GlobalSolution=zeros(runtime,Dim);
for run=1:runtime
    evaluation=0;
    evaluationNow=0;
    studentGroup=zeros(n,Dim);
    for i=1:n
        studentGroup(i,:)=rand(1,Dim).*(Ud-Ld)+Ld;
    end
    studentHistory=studentGroup;
    studentObjVal=fobj(studentGroup,ProblemStruct);
    evaluationNow=evaluationNow+n;
    studentHistoryObjVal=studentObjVal;
    [minStudentObjVal,minInd]=min(studentHistoryObjVal);
    bestStudent=studentGroup(minInd,:);
    globalStudentObjVal=minStudentObjVal;
    globalStudent=bestStudent;
    historyBestQuality=globalStudent;
    fitness=calculateFitness(studentObjVal);
    GlobalMins(run,evaluation+1:evaluationNow)=minStudentObjVal;
    evaluation=evaluationNow;
    iter=0;
    while (evaluation < maxEvaluation)    %%%%% start iterations
        if (iter==0)
            dLight=doReadingfitnessDistanceBalanceV2( studentGroup, studentHistoryObjVal );
            [descLight,descInd]=sort(dLight,'descend');%
        else
            if( mod(iter, 10)==0)
                dLight=doReadingfitnessDistanceBalanceV2( studentGroup, studentHistoryObjVal );
                [descLight,descInd]=sort(dLight,'descend');
            end
        end
        alpha=(aa-(evaluation/maxEvaluation)*(aa-bb));
        alpha=1-alpha;
        fixN=fix(alpha*n);
        %%exploitation process
        for i=1:fixN
            studyDif=globalW*rand().*(globalStudent-studentGroup(descInd(i),:))+historyW*rand().*(studentHistory(descInd(i),:)-studentGroup(descInd(i),:));
            sol=studentGroup(descInd(i),:)+studyDif;
            ind=find(sol<Ld);
            sol(ind)=Ld(ind);
            ind=find(sol>Ud);
            sol(ind)=Ud(ind);
            ObjValSol=fobj(sol,ProblemStruct);
            FitnessSol=calculateFitness(ObjValSol);
            evaluationNow=evaluationNow+1;
            studentGroup(descInd(i),:)=sol;
            studentObjVal(descInd(i))=ObjValSol;
            fitness(descInd(i))=FitnessSol;
            descLight(i)=FitnessSol;
            % /*a greedy selection is applied between the current solution i and its mutant*/
            if (ObjValSol<studentHistoryObjVal(descInd(i)))
                studentHistory(descInd(i),:)=sol;
                studentHistoryObjVal(descInd(i))=ObjValSol;
                if studentHistoryObjVal(descInd(i))<globalStudentObjVal
                    globalStudentObjVal=studentHistoryObjVal(descInd(i));
                    globalStudent=studentHistory(descInd(i),:);

                end
            end
        end
        index = fitnessDistanceBalance( studentGroup, studentObjVal );
        DglobalPara=studentGroup(index,:);
        %%%%exploration process
        for i=(fixN+1):n
            for j=1:Dim
                a=fix(rand*(n))+1;
                while(a==i)
                    a=fix(rand*n)+1;
                end
                b=fix(rand*n)+1;
                while((b==i)||(b==a))
                    b=fix(rand*n)+1;
                end

                tmpJ=(studentGroup(a,j)-studentGroup(b,j));

                studentGroup(descInd(i),j)= globalStudent(j)+rand()*tmpJ;
            end
            sol=studentGroup(descInd(i),:);
            % boundary
            ind=find(sol<Ld);
            sol(ind)=Ld(ind);
            ind=find(sol>Ud);
            sol(ind)=Ud(ind);
            ObjValSol=fobj(sol,ProblemStruct);
            FitnessSol=calculateFitness(ObjValSol);
            evaluationNow=evaluationNow+1;
            studentGroup(descInd(i),:)=sol;
            studentObjVal(descInd(i))=ObjValSol;
            fitness(descInd(i))=FitnessSol;
            descLight(i)=FitnessSol;
            if (ObjValSol<studentHistoryObjVal(descInd(i)))
                studentHistory(descInd(i),:)=sol;
                studentHistoryObjVal(descInd(i))=ObjValSol;
                if studentHistoryObjVal(descInd(i))<globalStudentObjVal
                    globalStudentObjVal=studentHistoryObjVal(descInd(i));
                    globalStudent=studentHistory(descInd(i),:);
                end
            end

        end%%end of exploration

        if abs(globalStudentObjVal-historyBestQuality)/abs(historyBestQuality)<=errorVal
            count=count+1;
        else
            count=0;
        end
        historyBestQuality=globalStudentObjVal;
        if count>SemesterLength
            for i=1:n
                studentGroup(i,:)=(ub-lb)*rand+lb;
                studentObjVal(i)=fobj(studentGroup(i,:),ProblemStruct);
                evaluationNow=evaluationNow+1;
                if studentObjVal(i)<studentHistoryObjVal(i)
                    studentHistoryObjVal(i)=studentObjVal(i);
                    studentHistory(i,:)=studentGroup(i,:);
                    if studentHistoryObjVal(i)<globalStudentObjVal
                        globalStudentObjVal=studentHistoryObjVal(i);
                        globalStudent=studentHistory(i,:);
                    end
                end
            end
            count=0;
            historyBestQuality=globalStudentObjVal;
        end

        [minStudentObjVal,minInd]=min(studentHistoryObjVal);
        bestStudent=studentGroup(minInd,:);
        if(minStudentObjVal<globalStudentObjVal)
            globalStudentObjVal=minStudentObjVal;
            globalStudent=bestStudent;
        end
        if(evaluationNow<maxEvaluation)
            GlobalMins(run,evaluation+1:evaluationNow)=globalStudentObjVal;
            evaluation=evaluationNow;
        else
            GlobalMins(run,evaluation+1:maxEvaluation)=globalStudentObjVal;
            break;
        end
        GlobalSolution(run,:)=globalStudent;
        iter=iter+1;
    end%%%% end of iterations
    fprintf('runs=%d iter=%d studentObjVal=%g\n',run,iter,minStudentObjVal);
end%%end of runs






