function [r,eta,sigma]=parameters(indices)
    %%%%%%table 5.2.2.1-4 de 3GPP TS 38.214 version 16.5.0
    efficacite=[0.0586 0.0977 0.1523 0.2344 0.3770 0.6016 0.8770 1.1758 1.4766 1.9141 2.4063 2.7305 3.3223 3.9023 4.5234];
    F=3072;%bit/paquet %size update, how many bite<7update
    RB=F./efficacite/26; 

    for l=1:length(indices)
        r(l)=RB(indices(l));
    end

    eta=ones(1,length(r))*3;
    sigma=ones(1,length(r));
end