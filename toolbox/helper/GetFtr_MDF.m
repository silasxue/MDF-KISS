%% Ftr
load('./data/viper_features_full.mat'); % Original features provided by KISSME
if exist('./data/viper_features_MDF.mat')
    load('./data/viper_features_MDF.mat');
    return;
end;
trainidx=1:2:size(X,2); % Half the dataset
d1=75;dm=535;

%% 1st PCA
Xt=X(:,trainidx);
muXt=mean(Xt,2);
[U1,Yt,D1]= pca(Xt');
Yt=Yt';
Y=U1'*(X-repmat(muXt,[1,size(X,2)]));

%% MDF
tic;
Ytm=Yt(1:dm,:);
Ym=Y(1:dm,:);
muYtm=mean(Ytm');
covYtm=cov(Ytm');
inv_covYtm=inv(covYtm);
MDF=zeros(1,size(Ym,2));
for i=1:size(Ym,2)
    MDF(i)=(Ym(:,i)-muYtm')'*inv_covYtm*(Ym(:,i)-muYtm');
end
Y=[Y(1:d1,:);MDF];
%% 2nd PCA
Yt=Y(:,trainidx);
muYt=mean(Yt,2);
[U2,Zt,D2]= pca(Yt');
Z=U2'*(Y-repmat(muYt,[1,size(Y,2)]));
toc;
ux2=Z;
save('./data/viper_features_MDF.mat','ux2');
