%% PlotAcc
% Function to plot accuracy. No inputs, as it is run from the NN_master.m
% file
figure; hold on
plot(25.*[(1:1:length(testAccM))],testAccM)
xlabel('Epochs')
ylabel('Test Accuracy (%)')
grid on

figure; hold on
plot(25.*[(6:1:length(testAccM))],testAccM(6:end))
xlabel('Epochs')
ylabel('Test Accuracy (%)')
grid on
set(gca, 'YScale', 'log')