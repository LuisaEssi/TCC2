fs = 512;
h_emg = load('novo.txt');

values_sred = novo(:,1);
values_sir = novo(:,2);
values_red = novo(:,3);
values_ir = novo(:,4);

t = linspace(0, length(values_ir)/fs, length(values_ir));

figure
plot(t,values_ir, 'LineWidth', 1)
hold on
plot(t,values_red, 'LineWidth', 2)
legend('Sinal original ir', 'Sinal original red')
xlabel('Tempo (s)')
ylabel('Amplitude')
