figure

data = %add your dataset here

bar(unique(data),histc(data,unique(data)))
hold on

h=histfit(data,5);
set(h(2),'color','r')
delete(h(1))
x=unique(data);
y=histc(data,unique(data));
ylim([0,1+max(y)]);
for i1=1:numel(y)
    text(x(i1),y(i1),num2str(y(i1)),...
               'HorizontalAlignment','center',...
               'VerticalAlignment','bottom', 'FontSize', 20)
end
title('Histogram fittet in a Normal Distribution (n=30)', 'FontSize', 20)
xlabel('Score', 'FontSize', 20)
ylabel('Participants', 'FontSize', 20)
