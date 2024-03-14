
function delay= randomdelay(userconnect,RBnumber,time)

% clear all
% usernumber=3;
% RBnumber=3;
% d=rand(usernumber,1)*500;  
% 
% I=([0.06 0.1 0.14]-0.04)*0.001;
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
% userconnect=[0, 0,1];
numberofuser=length(find(userconnect==1));



allocationscheme=randperm(RBnumber,numberofuser);
a=zeros(1,numberofuser);
ccc=find(userconnect==1);
for i=1:1:numberofuser
   a(1,i)=time(ccc(1,i),allocationscheme(1,i));
    
end





delay=max(a);








