% this code assumes dataN_aurco.mat files are available that include 10
% AUROC values for different network sizes. It then plots boxplots for the
% results.
load('data10_auroc.mat')
boxplot(auroc)
hold on

load('data50_auroc.mat')
boxplot(auroc,'Positions',1.5)
hold on

load('data100_auroc.mat')
boxplot(auroc,'Positions',2)
hold on

load('data150_auroc.mat')
boxplot(auroc,'Positions',2.5)
hold on

load('data200_auroc.mat')
boxplot(auroc,'Positions',3)
hold on

load('data1000_auroc.mat')
boxplot(auroc,'Positions',3.5)
hold on

load('datafull_auroc.mat')
boxplot(auroc,'Positions',4)
hold on

ylim([0.5,1])
set(gca, 'XTick', 1:0.5:4);
set(gca, 'XTickLabel', {'|V|=10,|E|=69', '|V|=50,|E|=1355','|V|=100,|E|=3059','|V|=150,|E|=4335','|V|=200,|E|=5473','|V|=1000,|E|=24240','|V|=17735,|E|=393970'});  % Set tick labels
set(gca, 'FontSize', 14); 

ylabel('AUROC', 'FontSize', 14);


