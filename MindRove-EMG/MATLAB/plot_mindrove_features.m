function fig = plot_mindrove_features(sess, options)
%PLOT_MINDROVE_FEATURES  Plot sync signals with decode features from mindrove band
%
% Syntax:
%   fig = plot_mindrove_features(sess,'Name',value,...);
%
% Example:
%   subj="max";
%   yyyy = 2024;
%   mm = 11;
%   dd = 3;
%   block=1;
%   sess = sprintf("%s_%04d_%02d_%02d_%d", subj, yyyy, mm, dd, block);
%   fig = plot_mindrove_features(sess);
%
% Inputs:
%   sess {mustBeTextScalar} % e.g. "max_2024_11_02_1"
%
% Options:
%   InputRoot = strrep(fullfile(pwd,"../data"),"\","/");
%   ColorOrder (:,3) double {mustBeInRange(options.ColorOrder,0,1)} = turbo(8);
%
% Output:
%   fig - Figure handle with sync signals and 8-channel EMG with timestamps
%
% See also:
%   read_mindrove, MindRoveCode, parse_mindrove_markers

arguments
    sess {mustBeTextScalar} % e.g. "max_2024_11_02_1"
    options.InputRoot = strrep(fullfile(pwd,"../data"),"\","/");
    options.ColorOrder (:,3) double {mustBeInRange(options.ColorOrder,0,1)} = turbo(8);
end

[x, ~, C] = read_mindrove(sprintf("%s/%s.tsv",options.InputRoot,sess)); 
fig = figure('Color','w','Name','MindRove sync signals', ...
    'Position', [365   115   560   700], ...
    'WindowState', 'maximized'); 
n_rows = 3;
L = tiledlayout(fig, n_rows, 1);

ax = gobjects(n_rows,1);
ax(1) = nexttile(L);
set(ax(1),'NextPlot','add','ColorOrder',options.ColorOrder, ...
    'YTick', 0:7, 'YLim', [-0.1, 8.1]);
plot(ax(1), C.dt, table2array(C(:,3:10)) ./ 5 + (0:7), ...
    'LineWidth', 1.5);
% legend(ax(1),C.Properties.VariableNames(3:10), ...
%     'Location', 'north', 'Orientation', 'horizontal', ...
%     'NumColumns', 4);
ylabel(ax(1), "Feature");

ax(2) = nexttile(L);
yt = unique(int32(C.decode));
set(ax(2),'NextPlot','add','YTick',yt,'YTickLabel',string(MindRoveCode(yt)));
plot(ax(2),C.dt, C.prompt,'Color','b', 'LineWidth', 2.5);
plot(ax(2),C.dt, C.decode,'Color','m', 'LineStyle', ':');
ylabel(ax(2),"\color{blue}Prompt/\color{magenta}Decode \color{black}Value");

ax(3) = nexttile(L);
yt = unique(int32(x.sync));
set(ax(3),'NextPlot','add','YTick',yt,'YTickLabel',string(MindRoveCode(yt)));
plot(ax(3),x.t, x.sync,'Color','k'); 
ylabel(ax(3),"Marker Value");

linkaxes(ax,'x');

end