
%% ��ջ�������
close all;
clear all;
clc;
format compact;
%% ���ݵ���ȡ��Ԥ����
data=xlsread('ɸѡ������');
ts =  data((1:320),1);%ѵ�������
tsx = data((1:320),2:end);%ѵ��������
tts=data((321:end),1);%Ԥ�⼯���
ttx= data((321:end),2:end);%Ԥ�⼯����
% ����Ԥ����,��ԭʼ���ݽ��й�һ��
ts = ts';
tsx = tsx';
tts=tts';
ttx=ttx';

% mapminmaxΪmatlab�Դ���ӳ�亯��	
% ��ts���й�һ��
[TS,TSps] = mapminmax(ts,-1,1);	%ʸ����һ��
[TTS,TTSps]= mapminmax(tts,-1,1);
TS = TS';
TTS=TTS';

% mapminmaxΪmatlab�Դ���ӳ�亯��
% ��tsx���й�һ��
[TSX,TSXps] = mapminmax(tsx,-1,1);	%����ֵ��һ��
[TTX,TTXps] = mapminmax(ttx,-1,1);	
% ��TSX����ת��,�Է���libsvm����������ݸ�ʽҪ��
TSX = TSX';
TTX = TTX';

%% ѡ��ع�Ԥ�������ѵ�SVM����c&g

% ���в���ѡ��: 
[bestmse,bestc,bestg] = SVMcgForRegress(TS,TSX,-10,10,-10,10);
% ��ӡ����ѡ����
disp('��ӡ����ѡ����');
str = sprintf( 'Best Cross Validation MSE = %g Best c = %g Best g = %g',bestmse,bestc,bestg);
disp(str);


%% ���ûع�Ԥ�������ѵĲ�������SVM����ѵ��
cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg) , ' -s 3 -p 0.01'];
model = svmtrain(TS,TSX,cmd);

%% SVM����ع�Ԥ��
[predict,mse] = svmpredict(TS,TSX,model);
[predict_2,mse_2] = svmpredict(TTS,TTX,model);
predict = mapminmax('reverse',predict',TSps);
predict_2 = mapminmax('reverse',predict_2',TTSps);
predict = predict';
predict_2 =predict_2'

% ������������
N = length(tts);
RMSE = sqrt((sum((tts-predict_2').^2))/N)
% % ���ϵ��
% N = length(tts);
% YUCE_R2 = (N*sum(predict_2'.*tts)-sum(predict_2)*sum(tts))^2/((N*sum((predict_2).^2)-(sum(predict_2'))^2)*(N*sum((tts).^2)-(sum(tts))^2))
%% ������������Լ���
figure;
plot(tts,'-o');
hold on;
plot(predict_2,'r-^');
legend('ʵ�ʸ���','Ԥ�⸺��');
hold off;
title('SVMԤ�����ͼ','FontSize',12);
xlabel('2019��11��20��-2019��12��30��','FontSize',12);
ylabel('���ɣ�KW��','FontSize',12);