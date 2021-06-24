
%% 清空环境变量
close all;
clear all;
clc;
format compact;
%% 数据的提取和预处理
data=xlsread('筛选后数据');
ts =  data((1:320),1);%训练集输出
tsx = data((1:320),2:end);%训练集输入
tts=data((321:end),1);%预测集输出
ttx= data((321:end),2:end);%预测集输入
% 数据预处理,将原始数据进行归一化
ts = ts';
tsx = tsx';
tts=tts';
ttx=ttx';

% mapminmax为matlab自带的映射函数	
% 对ts进行归一化
[TS,TSps] = mapminmax(ts,-1,1);	%矢量归一化
[TTS,TTSps]= mapminmax(tts,-1,1);
TS = TS';
TTS=TTS';

% mapminmax为matlab自带的映射函数
% 对tsx进行归一化
[TSX,TSXps] = mapminmax(tsx,-1,1);	%特征值归一化
[TTX,TTXps] = mapminmax(ttx,-1,1);	
% 对TSX进行转置,以符合libsvm工具箱的数据格式要求
TSX = TSX';
TTX = TTX';

%% 选择回归预测分析最佳的SVM参数c&g

% 进行参数选择: 
[bestmse,bestc,bestg] = SVMcgForRegress(TS,TSX,-10,10,-10,10);
% 打印参数选择结果
disp('打印参数选择结果');
str = sprintf( 'Best Cross Validation MSE = %g Best c = %g Best g = %g',bestmse,bestc,bestg);
disp(str);


%% 利用回归预测分析最佳的参数进行SVM网络训练
cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg) , ' -s 3 -p 0.01'];
model = svmtrain(TS,TSX,cmd);

%% SVM网络回归预测
[predict,mse] = svmpredict(TS,TSX,model);
[predict_2,mse_2] = svmpredict(TTS,TTX,model);
predict = mapminmax('reverse',predict',TSps);
predict_2 = mapminmax('reverse',predict_2',TTSps);
predict = predict';
predict_2 =predict_2'

% 均方根误差计算
N = length(tts);
RMSE = sqrt((sum((tts-predict_2').^2))/N)
% % 相关系数
% N = length(tts);
% YUCE_R2 = (N*sum(predict_2'.*tts)-sum(predict_2)*sum(tts))^2/((N*sum((predict_2).^2)-(sum(predict_2'))^2)*(N*sum((tts).^2)-(sum(tts))^2))
%% 结果分析（测试集）
figure;
plot(tts,'-o');
hold on;
plot(predict_2,'r-^');
legend('实际负荷','预测负荷');
hold off;
title('SVM预测输出图','FontSize',12);
xlabel('2019年11月20日-2019年12月30日','FontSize',12);
ylabel('负荷（KW）','FontSize',12);