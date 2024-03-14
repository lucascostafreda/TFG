function mindelay= mindelay(userconnect,RBnumber,time)

% clear all
% usernumber=3;
% RBnumber=5;
% %numberofuser=2;
% d=rand(usernumber,1)*500;  
% 
% I=([0.06 0.08 0.1 0.14 0.16]-0.04)*0.001;
% 
% P=1;
% 
% SINR=P*1*(d(1:usernumber,1).^(-2))./I;
% 
% 
% rate=log2(1+SINR);
% 
% time=39760*32./rate/1024/1024;
% 
% userconnect=[1,1,1];


numberofuser=length(find(userconnect==1));
usernumber=length(userconnect);

%selectuser=nchoosek([1:1:usernumber],numberofuser);

%d=rand(usernumber,1)*500;  

%I=([0.06 0.1 0.14]-0.04)*0.001;






f=[zeros(usernumber*RBnumber,1);1];
ub = [ones(usernumber*RBnumber,1);Inf];
lb = zeros(usernumber*RBnumber+1,1);
intcon = 1:RBnumber*usernumber;

%%%%%%%%%%% set A
A=zeros(usernumber+RBnumber,usernumber*RBnumber+1);
options = optimoptions('intlinprog','Display','off');
for i=0:1:usernumber
    if i==0
        for j=1:1:usernumber
         A(1:RBnumber,(j-1)*RBnumber+1:j*RBnumber)=eye(RBnumber:RBnumber);
        end
    else
        A(i+RBnumber,(i-1)*RBnumber+1:i*RBnumber)=time(i,:);
        A(i+RBnumber,RBnumber*usernumber+1)=-1;
    end
end
b=ones(usernumber+RBnumber,1);
b(RBnumber+1:usernumber+RBnumber)=0;


%%%%%%%%%%%%%%%% set Aeq and beq %%%%%%%%%%%%%%%%%%%%%%%%
 beq=ones(numberofuser,1);

%for k=1:1:length(selectuser(:,1))

Aeq=zeros(numberofuser,RBnumber*usernumber+1);
j=1;
ccc=find(userconnect==1);
for i=1:1:numberofuser
   
     Aeq(i,(ccc(1,i)-1)*RBnumber+1:ccc(1,i)*RBnumber)=1;
    
end

% Aeq=[1 1 1 0 0 0 0 0 0 0
%      0 0 0 1 1 1 0 0 0 0
%      0 0 0 0 0 0 1 1 1 0];
 

x=intlinprog(f,intcon,A,b,Aeq,beq,lb,ub,options);
mindelay=x(usernumber*RBnumber+1,1);

%end

