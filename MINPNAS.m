
clear all


datanumber=1000;
alpha=0.85;

k=0;

numberofneuron=50;
averagenumber=1;
iteration=200;


I=([0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15]-0.04)*0.001;

RBnumber=length(I);

P=1;


wlength=50*784;
lwlength=500;
blength=50;
oblength=10;


[trainingdata, traingnd] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
trainingdata = double(reshape(trainingdata, size(trainingdata,1)*size(trainingdata,2), []).');
trainingdata=double(trainingdata);

traingnd = double(traingnd);
traingnd(traingnd==0)=10;
traingnd=dummyvar(traingnd); 





[testdata, testgnd] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');
testdata = double(reshape(testdata, size(testdata,1)*size(testdata,2), []).');
testgnd = double(testgnd);
testgnd(testgnd==0)=10;

kk=0;


averageerror=[];
averagedelay=[];
numberofitetation=[];



%%%%%%%%%%%%%%%%%%%%%%%% coding setting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
v_fQRate = [1, 2];
v_nQuantizaers   = [...          % Curves
    0 ...                   % Dithered 3-D lattice quantization 
    1 ...                   % Dithered 2-D lattice quantization    
    1 ...                   % Dithered scalar quantization      
    1 ...                   % QSGD 
    1 ...                   % Uniform quantization with random unitary rotation    
    1 ...                   % Subsampling with 3 bits quantizers
    ];

global gm_fGenMat2D;
global gm_fLattice2D;
% Clear lattices
gm_fGenMat2D = [];
gm_fLattice2D = [];
% Do full search over the lattice
stSettings.OptSearch = 1;

s_fRate=4;
stSettings.type =2;
stSettings.scale=2;




usernumber=15;
%numberofuser=6;

proposedFL=1;
proposedUA=0;
baseline2=0;
baseline3=0;



for userno=10:2:10
    
    kk=kk+1;
    
    numberofuser=userno;
   % usernumber=userno; 
%     if usernumber==3
%      numberofuser=3;
%     else 
%       numberofuser=5;  
%     end
    


for average=1:1:averagenumber
probability4=[];
timea=[];
delay=[];
mi=[];    
d=rand(usernumber,1)*500;    

% userconnectvector=ones(usernumber,1);
% userconnectvector(1:usernumber-numberofuser)=0;
% aa=perms(userconnectvector);
% userconnectvector1=unique(aa,'rows');
% minimumdelay=zeros(length(userconnectvector1(:,1)),1);

error=[];
    
    
    SINR=P*1*(d(1:usernumber,1).^(-2))./I;


rate=2*log2(1+SINR);

time=39760*32./rate/1024/1024;


iterationtime=[];
iterationtimerandom=[];
userupdate=zeros(iteration,usernumber);
userupdate2=[1:1:usernumber];   
wupdate=zeros(iteration,usernumber);   % local model for each user




w=[];
lw=[];
b=[];

ob=[];

cc=[];
wnew=zeros(numberofneuron,usernumber);
lwnew=zeros(numberofneuron,usernumber);
bnew=zeros(numberofneuron,usernumber);

obnew=zeros(1,usernumber);


wglobal=[];            % global training model
lwglobal=[];            % global training model
bglobal=[];  
obglobal=[]; 
deviationw=[];
deviationlw=[];
deviationb=[];
deviationob=[];

 net1 = patternnet(numberofneuron);                   % each linear network for each user
  
 % net1.trainFcn = 'traingd';
    net1.divideFcn = '';
 net1.inputs{1}.processFcns={};
 net1.outputs{2}.processFcns={};
net1.trainParam.epochs = 1;
net1.trainParam.showWindow = 0;
input1=[];
output1=[];
account1=0;


 net2 = patternnet(numberofneuron); 
  
  %net2.trainFcn = 'traingd';
    net2.divideFcn = '';
      net2.inputs{1}.processFcns={};
 net2.outputs{2}.processFcns={};
net2.trainParam.epochs = 1;
net2.trainParam.showWindow = 0;
input2=[];
output2=[];
account2=0;



 net3 = patternnet(numberofneuron); 
  
  %net3.trainFcn = 'traingd';
    net3.divideFcn = '';
      net3.inputs{1}.processFcns={};
 net3.outputs{2}.processFcns={};
net3.trainParam.epochs = 1;
net3.trainParam.showWindow = 0;
input3=[];
output3=[];
account3=0;


if usernumber>3

 net4 = patternnet(numberofneuron); 
  
 % net4.trainFcn = 'traingd';
    net4.divideFcn = '';
      net4.inputs{1}.processFcns={};
 net4.outputs{2}.processFcns={};
net4.trainParam.epochs = 1;
net4.trainParam.showWindow = 0;
input4=[];
output4=[];
account4=0;



net5 = patternnet(numberofneuron);
 
  %net5.trainFcn = 'traingd';
    net5.divideFcn = ''; 
  net5.inputs{1}.processFcns={};
 net5.outputs{2}.processFcns={};
net5.trainParam.epochs = 1;
net5.trainParam.showWindow = 0;
input5=[];
output5=[];
account5=0;




net6 = patternnet(numberofneuron);
   net6.divideFcn = '';
  %net6.trainFcn = 'traingd';
      net6.inputs{1}.processFcns={};
 net6.outputs{2}.processFcns={};
net6.trainParam.epochs = 1;
net6.trainParam.showWindow = 0;
input6=[];
output6=[];
account6=0;


if usernumber>6


net7 = patternnet(numberofneuron);
   net7.divideFcn = '';
  %net7.trainFcn = 'traingd';
      net7.inputs{1}.processFcns={};
 net7.outputs{2}.processFcns={};
net7.trainParam.epochs = 1;
net7.trainParam.showWindow = 0;
input7=[];
output7=[];
account7=0;

net8 = patternnet(numberofneuron);
   net8.divideFcn = '';
  %net8.trainFcn = 'traingd';
      net8.inputs{1}.processFcns={};
 net8.outputs{2}.processFcns={};
net8.trainParam.epochs = 1;
net8.trainParam.showWindow = 0;
input8=[];
output8=[];
account8=0;


net9 = patternnet(numberofneuron);
   net9.divideFcn = '';
%  net9.trainFcn = 'traingd';
      net9.inputs{1}.processFcns={};
 net9.outputs{2}.processFcns={};
net9.trainParam.epochs = 1;
net9.trainParam.showWindow = 0;
input9=[];
output9=[];
account9=0;


if usernumber>9

net10 = patternnet(numberofneuron);
   net10.divideFcn = '';
  %net10.trainFcn = 'traingd';
      net10.inputs{1}.processFcns={};
 net10.outputs{2}.processFcns={};
net10.trainParam.epochs = 1;
net10.trainParam.showWindow = 0;
input10=[];
output10=[];
account10=0;


net11 = patternnet(numberofneuron);
  net11.divideFcn = '';
  %net11.trainFcn = 'traingd';
      net11.inputs{1}.processFcns={};
 net11.outputs{2}.processFcns={};
net11.trainParam.epochs = 1;
net11.trainParam.showWindow = 0;
input11=[];
output11=[];
account11=0;

net12 = patternnet(numberofneuron);
 net12.divideFcn = '';
  %net12.trainFcn = 'traingd';
      net12.inputs{1}.processFcns={};
 net12.outputs{2}.processFcns={};
net12.trainParam.epochs = 1;
net12.trainParam.showWindow = 0;
input12=[];
output12=[];
account12=0;


if usernumber>12

net13 = patternnet(numberofneuron);
  net13.divideFcn = '';
  %net13.trainFcn = 'traingd';
      net13.inputs{1}.processFcns={};
 net13.outputs{2}.processFcns={};
net13.trainParam.epochs = 1;
net13.trainParam.showWindow = 0;
input13=[];
output13=[];
account13=0;

net14 = patternnet(numberofneuron);
  net14.divideFcn = '';
  %net14.trainFcn = 'traingd';
      net14.inputs{1}.processFcns={};
 net14.outputs{2}.processFcns={};
net14.trainParam.epochs = 1;
net14.trainParam.showWindow = 0;
input14=[];
output14=[];
account14=0;


net15 = patternnet(numberofneuron);
  net15.divideFcn = '';
  %net15.trainFcn = 'traingd';
      net15.inputs{1}.processFcns={};
 net15.outputs{2}.processFcns={};
net15.trainParam.epochs = 1;
net15.trainParam.showWindow = 0;
input15=[];
output15=[];
account15=0;


end
end
end
end



bb=[];


for i=1:1:iteration


for user=1:1:usernumber
  
        x1=trainingdata((user-1)*datanumber+1:user*datanumber,:);
        y1=traingnd((user-1)*datanumber+1:user*datanumber,:);
    
    clear netvaluable;
    Winstr1=strcat('net',int2str(user));
     eval(['netvaluable','=',Winstr1,';']);
    
if i > 1

    netvaluable.IW{1,1}=wglobal;
    netvaluable.LW{2,1}=lwglobal;
     netvaluable.b{1,1}=bglobal;
     netvaluable.b{2,1}=obglobal;   
end


oldnetvaluable=netvaluable;

[netvaluable,tr] =  train(netvaluable,x1',y1');

if i==1
       wglobal=zeros(size(netvaluable.IW{1,1}));
    lwglobal=zeros(size(netvaluable.LW{2,1}));
    bglobal=zeros(size(netvaluable.b{1,1}));
    obglobal=zeros(size(netvaluable.b{2,1}));
end





w(:,:,user)=netvaluable.IW{1,1};

lw(:,:,user)=netvaluable.LW{2,1};

b(:,:,user)=netvaluable.b{1,1};
ob(:,:,user)=netvaluable.b{2,1};


%%%%%%%%%%%%%%%%%%%%%%%%%%%%Proposead algorithm%%%%%%%%%%%%%%%%%%%%%%%%%%%
if baseline3~=1
if i==1    
deviationw(:,:,user)=netvaluable.IW{1,1};
deviationlw(:,:,user)=netvaluable.LW{2,1};
deviationb(:,:,user)=netvaluable.b{1,1};
deviationob(:,:,user)=netvaluable.b{2,1};

else
deviationw(:,:,user)=netvaluable.IW{1,1}-oldnetvaluable.IW{1,1};
deviationlw(:,:,user)=netvaluable.LW{2,1}-oldnetvaluable.LW{2,1};
deviationb(:,:,user)=netvaluable.b{1,1}-oldnetvaluable.b{1,1};
deviationob(:,:,user)=netvaluable.b{2,1}-oldnetvaluable.b{2,1};

oldw(:,:,user)=oldnetvaluable.IW{1,1};
oldlw(:,:,user)=oldnetvaluable.LW{2,1};
oldb(:,:,user)=oldnetvaluable.b{1,1};
oldob(:,:,user)=oldnetvaluable.b{2,1};

end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


eval([Winstr1,'=','netvaluable',';']);
end


if proposedFL==1
 
sumdw=sum(sum(sqrt(deviationw.^2),1));

sumdlw=sum(sum(sqrt(deviationlw.^2),1));

sumdb=sum(sum(sqrt(deviationb.^2),1));

sumdob=sum(sum(sqrt(deviationob.^2),1));

sumd=sumdw+sumdlw+sumdb+sumdob;

 [xx,dex]=sort(sumd);
%bb(i,:)=dex(usernumber-numberofuser+1:usernumber);

if  sum(sumd)>0
  probability=reshape(sumd/sum(sumd),1,usernumber);
     probabilityd=(max(d)-d)/sum(max(d)-d);
     
   probability=probability*alpha+(1-alpha)*probabilityd';
else
    probability=ones(1,usernumber)/usernumber;
end
     vector=[1:1:usernumber];
     for llll=1:1:numberofuser

    bb(i,llll)=randsrc(1,1,[vector; probability]);
    
    position=find(vector==bb(i,llll));
    probability1=probability(position);
    probability(position)=[];
    vector(position)=[];
    probability=probability./(1-probability1);
    %end
     end

% end
 
 
 
   

%userupdate(i,bb(i,:))=1;


%

for code=1:1:numberofuser
    
  
    
    
    wvector=reshape(deviationw(:,:,bb(i,code)),[wlength,1]);

    lwvector=reshape(deviationlw(:,:,bb(i,code)),[lwlength,1]);

   
    m_fH1 = [wvector;lwvector;deviationb(:,:,bb(i,code));deviationb(:,:,bb(i,code))]; 
%    
   [m_fHhat1, ~] = m_fQuantizeData(m_fH1, s_fRate, stSettings); % coding and decoding
   
%    deviationw(:,:,bb(i,code))=reshape(m_fHhat1(1:wlength),[50,784]);
%   deviationlw(:,:,bb(i,code))=reshape(m_fHhat1(wlength+1:lwlength+wlength),[10,50]);
%   deviationb(:,:,bb(i,code))=m_fHhat1(lwlength+wlength+1:lwlength+wlength+blength);
%   deviationob(:,:,bb(i,code))=m_fHhat1(lwlength+wlength+blength+1:lwlength+wlength+blength+oblength);
    
    if i==1
   
     w(:,:,bb(i,code))=reshape(m_fHhat1(1:wlength),[50,784]);
     lw(:,:,bb(i,code))=reshape(m_fHhat1(wlength+1:lwlength+wlength),[10,50]);
     b(:,:,bb(i,code))=m_fHhat1(lwlength+wlength+1:lwlength+wlength+blength);
     ob(:,:,bb(i,code))=m_fHhat1(lwlength+wlength+blength+1:lwlength+wlength+blength+oblength);  
    else
     w(:,:,bb(i,code))=oldw(:,:,bb(i,code))+reshape(m_fHhat1(1:wlength),[50,784]);
     lw(:,:,bb(i,code))=oldlw(:,:,bb(i,code))+reshape(m_fHhat1(wlength+1:lwlength+wlength),[10,50]);
     b(:,:,bb(i,code))=oldb(:,:,bb(i,code))+m_fHhat1(lwlength+wlength+1:lwlength+wlength+blength);
     ob(:,:,bb(i,code))=oldob(:,:,bb(i,code))+m_fHhat1(lwlength+wlength+blength+1:lwlength+wlength+blength+oblength);  
    end
    
end


  wglobal=1/numberofuser*sum(w(:,:,bb(i,:)),3);  % global training model
  lwglobal=1/numberofuser*sum(lw(:,:,bb(i,:)),3);  % global training model
  bglobal=1/numberofuser*sum(b(:,:,bb(i,:)),3);
  obglobal=1/numberofuser*sum(ob(:,:,bb(i,:)),3);
 
 
  
 %%%%%%%%%%%%%%%%%%%%%%% Delay calculation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 userconnectvector=zeros(1,usernumber);
 userconnectvector(1,bb(i,:))=1;
  
 %delayposition=find(ismember(userconnectvector1,userconnectvector,'rows')==1);

 %if minimumdelay(delayposition,1)>0
 %    iterationtime(i)=minimumdelay(delayposition,1);  
 %else 
     iterationtime(i)=mindelay(userconnectvector,RBnumber,time);
 %    minimumdelay(delayposition,1)=iterationtime(i);
 %end
 
 iterationtimerandom(i)=randomdelay(userconnectvector,RBnumber,time);
 
 %[c, ia, ib] = intersect(selectuser,sort(bb(i,:)),'rows');

%iterationtime(i)=delay(ia,1);
 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 
 
 if proposedUA==1
 

     
   probability=ones(1,usernumber)/usernumber;

     
     vector=[1:1:usernumber];
     for llll=1:1:numberofuser

    bb(i,llll)=randsrc(1,1,[vector; probability]);
    
    position=find(vector==bb(i,llll));
    probability1=probability(position);
    probability(position)=[];
    vector(position)=[];
    probability=probability./(1-probability1);
    %end
     end

% end
 
 
 
   

%userupdate(i,bb(i,:))=1;


%

for code=1:1:numberofuser
    
  
    
    
    wvector=reshape(deviationw(:,:,bb(i,code)),[wlength,1]);

    lwvector=reshape(deviationlw(:,:,bb(i,code)),[lwlength,1]);

   
    m_fH1 = [wvector;lwvector;deviationb(:,:,bb(i,code));deviationb(:,:,bb(i,code))]; 
%    
   [m_fHhat1, ~] = m_fQuantizeData(m_fH1, s_fRate, stSettings); % coding and decoding
   
%    deviationw(:,:,bb(i,code))=reshape(m_fHhat1(1:wlength),[50,784]);
%   deviationlw(:,:,bb(i,code))=reshape(m_fHhat1(wlength+1:lwlength+wlength),[10,50]);
%   deviationb(:,:,bb(i,code))=m_fHhat1(lwlength+wlength+1:lwlength+wlength+blength);
%   deviationob(:,:,bb(i,code))=m_fHhat1(lwlength+wlength+blength+1:lwlength+wlength+blength+oblength);
    
    if i==1
   
     w(:,:,bb(i,code))=reshape(m_fHhat1(1:wlength),[50,784]);
     lw(:,:,bb(i,code))=reshape(m_fHhat1(wlength+1:lwlength+wlength),[10,50]);
     b(:,:,bb(i,code))=m_fHhat1(lwlength+wlength+1:lwlength+wlength+blength);
     ob(:,:,bb(i,code))=m_fHhat1(lwlength+wlength+blength+1:lwlength+wlength+blength+oblength);  
    else
     w(:,:,bb(i,code))=oldw(:,:,bb(i,code))+reshape(m_fHhat1(1:wlength),[50,784]);
     lw(:,:,bb(i,code))=oldlw(:,:,bb(i,code))+reshape(m_fHhat1(wlength+1:lwlength+wlength),[10,50]);
     b(:,:,bb(i,code))=oldb(:,:,bb(i,code))+m_fHhat1(lwlength+wlength+1:lwlength+wlength+blength);
     ob(:,:,bb(i,code))=oldob(:,:,bb(i,code))+m_fHhat1(lwlength+wlength+blength+1:lwlength+wlength+blength+oblength);  
    end
    
end


  wglobal=1/numberofuser*sum(w(:,:,bb(i,:)),3);  % global training model
  lwglobal=1/numberofuser*sum(lw(:,:,bb(i,:)),3);  % global training model
  bglobal=1/numberofuser*sum(b(:,:,bb(i,:)),3);
  obglobal=1/numberofuser*sum(ob(:,:,bb(i,:)),3);
 
 
  
 %%%%%%%%%%%%%%%%%%%%%%% Delay calculation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 userconnectvector=zeros(1,usernumber);
 userconnectvector(1,bb(i,:))=1;
  
% delayposition=find(ismember(userconnectvector1,userconnectvector,'rows')==1);

 %if minimumdelay(delayposition,1)>0
 %    iterationtime(i)=minimumdelay(delayposition,1);  
 %else 
     iterationtime(i)=mindelay(userconnectvector,RBnumber,time);
  %   minimumdelay(delayposition,1)=iterationtime(i);
 %end
 
 iterationtimerandom(i)=randomdelay(userconnectvector,RBnumber,time);
 
 %[c, ia, ib] = intersect(selectuser,sort(bb(i,:)),'rows');

%iterationtime(i)=delay(ia,1);
 
end

 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 

if baseline2==1
 
sumdw=sum(sum(sqrt(deviationw.^2),1));

sumdlw=sum(sum(sqrt(deviationlw.^2),1));

sumdb=sum(sum(sqrt(deviationb.^2),1));

sumdob=sum(sum(sqrt(deviationob.^2),1));

sumd=sumdw+sumdlw+sumdb+sumdob;

 [xx,dex]=sort(sumd);
%bb(i,:)=dex(usernumber-numberofuser+1:usernumber);

if sum(sumd)>0
  probability=reshape(sumd/sum(sumd),1,usernumber);
     probabilityd=(max(d)-d)/sum(max(d)-d);
     
   probability=probability*alpha+(1-alpha)*probabilityd';
else 
  probability= ones(1,usernumber)/usernumber;
end
     vector=[1:1:usernumber];
     for llll=1:1:numberofuser

    bb(i,llll)=randsrc(1,1,[vector; probability]);
    
    position=find(vector==bb(i,llll));
    probability1=probability(position);
    probability(position)=[];
    vector(position)=[];
    probability=probability./(1-probability1);
    %end
     end

  wglobal=1/numberofuser*sum(w(:,:,bb(i,:)),3);  % global training model
  lwglobal=1/numberofuser*sum(lw(:,:,bb(i,:)),3);  % global training model
  bglobal=1/numberofuser*sum(b(:,:,bb(i,:)),3);
  obglobal=1/numberofuser*sum(ob(:,:,bb(i,:)),3);
 
 %%%%%%%%%%%%%%%%%%%%%%% Delay calculation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 userconnectvector=zeros(1,usernumber);
 userconnectvector(1,bb(i,:))=1;
  
% delayposition=find(ismember(userconnectvector1,userconnectvector,'rows')==1);

 %if minimumdelay(delayposition,1)>0
  %   iterationtime(i)=minimumdelay(delayposition,1);  
 %else 
     iterationtime(i)=mindelay(userconnectvector,RBnumber,time);
 %    minimumdelay(delayposition,1)=iterationtime(i);
% end
 
 iterationtimerandom(i)=randomdelay(userconnectvector,RBnumber,time);
 
 %[c, ia, ib] = intersect(selectuser,sort(bb(i,:)),'rows');

%iterationtime(i)=delay(ia,1);
 
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%delay%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 

if baseline3==1
 

     
   probability=ones(1,usernumber)/usernumber;
 
     vector=[1:1:usernumber];
     for llll=1:1:numberofuser

    bb(i,llll)=randsrc(1,1,[vector; probability]);
    
    position=find(vector==bb(i,llll));
    probability1=probability(position);
    probability(position)=[];
    vector(position)=[];
    probability=probability./(1-probability1);
    %end
     end

  wglobal=1/numberofuser*sum(w(:,:,bb(i,:)),3);  % global training model
  lwglobal=1/numberofuser*sum(lw(:,:,bb(i,:)),3);  % global training model
  bglobal=1/numberofuser*sum(b(:,:,bb(i,:)),3);
  obglobal=1/numberofuser*sum(ob(:,:,bb(i,:)),3);
 
 %%%%%%%%%%%%%%%%%%%%%%% Delay calculation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 userconnectvector=zeros(1,usernumber);
 userconnectvector(1,bb(i,:))=1;
  
% delayposition=find(ismember(userconnectvector1,userconnectvector,'rows')==1);

 %if minimumdelay(delayposition,1)>0
  %   iterationtime(i)=minimumdelay(delayposition,1);  
 %else 
     iterationtime(i)=mindelay(userconnectvector,RBnumber,time);
 %    minimumdelay(delayposition,1)=iterationtime(i);
% end
 
 iterationtimerandom(i)=randomdelay(userconnectvector,RBnumber,time);
 
 %[c, ia, ib] = intersect(selectuser,sort(bb(i,:)),'rows');

%iterationtime(i)=delay(ia,1);
 
end

%delay(ia,1);










[nn,mm]=max(net1(testdata(1:10000,:)'));
    
    oo=mm'-testgnd(1:10000,:);
    
    error(i)=length(find(oo~=0))/10000;





%  if i>1 &isnan(sum(probability4(i,:)))
%          
%          break;
%          
% end



end





    

% finalerror(average,1)=error(i-errrornumber+1);
% averagedelay(average,1)=sum(iterationtime);
% numberofitetation(average,1)=i;

 %  if isnan(sum(probability4(i,:)))
 %   averageerror(average,kk)=error(i);    
 %  else
   averageerror(average,kk)=error(iteration);
 %  end
averagedelay(average,kk)=sum(iterationtime);
averagedelayrandom(average,kk)=sum(iterationtimerandom);
end

finalerror(kk,1)=sum(averageerror(:,kk))/averagenumber;
finaldelay(kk,1)=sum(averagedelay(:,kk))/averagenumber;
finaldelayrandom(kk,1)=sum(averagedelayrandom(:,kk))/averagenumber;
%finaliteration(kk,1)=sum(numberofitetation(:,kk))/averagenumber;
end



% 
mm(find(mm==10))=0;
% mm2=mm;

figure                                          % initialize figure
colormap(gray)                                  % set to grayscale
  
   a=105;
for i = 1:9                                    % preview first 36 samples
    subplot(3,3,i)                              % plot them in 6 x 6 grid
    digit = reshape(testdata(i+a, :), [28,28]);     % row = 28 x 28 image
    imagesc(digit)                              % show the image
    
    if i==1
        
    title('Proposed FL: 7')                   % show the label

    set(gca,'xtick',[]);
    set(gca,'ytick',[]);
    
    xlabel('Classical FL: 7');
    else 
          title(num2str(mm(i+a))) 
    %      xlabel(num2str(mm(i+a)));
          set(gca,'xtick',[]);
     set(gca,'ytick',[]);
    end
end




  xxdelay=[243.1422 343.1084 414.0626 493.8498 553.7755];
  xxdelaybaseline3=[249.4434, 362.9120, 439.9591, 515.8786, 581.3694];
  
  xxdelayrandom=[0.7069 1.0028 1.1773 1.3182 1.4200]*10^3;
  xxdelayrandombaseline3=[0.7262 1.0603 1.2325 1.3638 1.4628]*10^3;

  
% iteration=i;









%plot([-1:0.001:1],netap([-1:0.001:1]))


% figure(2)
% plot(wglobal)
% hold on
% plot(wglobal2)



% if k>2 && H(k)==H(k-1)
%     break
% end

%Time
%proposed algorithm   [142.23  371.8, 536.69 743.6546      612.13] 
%Classic  FL          [366.3   662.82 1129.3 1378.3        1500.6] 
                      %[827.8   1251   1227   1165.7        1245.7]

                      % 3       6       9        12        15   
%%%%Performanc       [0.1993  0.1472   0.1303   0.1205    0.119 ]
%%%%Performance      [0.1936  0.1507    0.142   0.1381    0.1370]


%Iteration            [ 
%Iteration            [94.25   114.595  110.7   106.6150   112.7] 


% 
% 
% aaa=[142.23  371.8, 536.69 743.6546      852.13]; 
% bbb=[366.3   662.82 1129.3 1378.3        1500.6];
% ccc=[3:3:15];
% 
%    figure (2) 
% hold on
% h1=plot(ccc,aaa,'k-s','LineWidth',3,'MarkerFaceColor','k','MarkerSize',6) ; 
% hold on
% h4=plot(ccc,bbb,'r--d','LineWidth',3,'MarkerFaceColor','r','MarkerSize',4) ; 
% hold on
% %h5=plot([3:1:7],h2new,' -g','LineWidth',1.6,'MarkerFaceColor','g','MarkerSize',6) ; 
% hh=legend([h1,h4],'Proposaed FL','Classic FL','Orientation','vertical');
% box on;
% axis([3,15,142.23,1550]);
% set(gca, 'XTick', [3:3:15]);
% grid on
% set(gca,'Fontsize',14);
% xlabel('Number of BSs');
% ylabel('Convergence time (s)');
% 
