function[Cindex1]=Cindex(Tpredict,Ttest,deltatest)
%-------INPUT---------%%%%%
% Tpredict------------=predicted time 
% Ttest---------------=observed data included censored data
% deltatest-----------=censored variable
%-----OUTPUT----------
% Cindex 
N=length(Ttest);
ConcordDeno=0;
ConcordNum=0;
for i=1:N
time1=Ttest(i);
timePred1=Tpredict(i);
for j=1:N
if (i~=j)
time2=Ttest(j);
timePred2=Tpredict(j);
ConcordNum=ConcordNum+(timePred2>timePred1)*(time2>time1)*(deltatest(i)==1)+(timePred2<timePred1)*(time2<time1)*(deltatest(j)==1)+0.5*((timePred2==timePred1)||(time2==time1))*(deltatest(i)==1)*(deltatest(j)==0)+0.5*((timePred2==timePred1)||(time2==time1))*(deltatest(j)==1)*(deltatest(i)==0);
ConcordDeno=ConcordDeno+(time2>time1)*(deltatest(i)==1)+(time2<time1)*(deltatest(j)==1)+(time2==time1)*(deltatest(i)==1)*(deltatest(j)==0)+(time2==time1)*(deltatest(i)==0)*(deltatest(j)==1);
end
end
end
Cindex1=ConcordNum/ConcordDeno;


