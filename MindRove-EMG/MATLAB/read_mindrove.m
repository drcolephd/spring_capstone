function [x, T, C, F] = read_mindrove(fname, options)
%READ_MINDROVE  Read mindrove data tsv format
%
% Example:
%   fname = "C:/MyRepos/Python/mindrove/data/arrows_2024_10_26_1.tsv";
%   [x, T, C, F] = read_mindrove(fname);
%
% Inputs:
%   fname - Name of the .tsv file logged by mindrove SDK.
%   
% Output:
%   x - Data struct with 'samples', etc. like other EMG loaders.
%   T - Table of tsv values directly
%   C - The parsed converted .csv produced from the binary logger that
%       tracks features, after recording has stopped.
%   F - The .csv exported (if it exists) during the direction prompt game.

arguments
    fname {mustBeTextScalar, mustBeFile}
    options.CutoffFrequency = 100; % Hz
    options.SampleRate = 500; % Hz
end

T = readtable(fname, ...
    'FileType', 'text', 'Delimiter', '\t');
tmp = table2array(T(:,1:8));
[b,a] = butter(3,options.CutoffFrequency / (options.SampleRate/2), 'high');
tmp = filter(b,a,tmp);
x.samples = tmp';
x.samples(:,1:15) = 0; % From `filter` use.
x.t = datetime(T.Var28,'ConvertFrom','posixtime')';
x.t.TimeZone = 'UTC';
x.t.TimeZone = 'America/New_York';
x.t = linspace(x.t(1), x.t(end), numel(x.t));

x.sync = parse_mindrove_markers(MindRoveCode(T.Var29'));
x.aux = T.Var20';
x.acc = table2array(T(:,21:26))';
x.counter = T.Var27';
x.battery = T.Var19';
if exist(strrep(fname,".tsv",".csv"),'file')==0
    C = [];
else
    C = readtable(strrep(fname,".tsv",".csv"));
    C.dt = C.date + C.time;
    C.dt.TimeZone = "America/New_York";
    C.decode = MindRoveCode(C.decode + 7);
    C.prompt = MindRoveCode(C.prompt + 7);
end
if exist(strrep(fname,".tsv","_features.csv"),'file')==0
    F = [];
else
    F = readtable(strrep(fname,".tsv", "_features.csv"));
    F.label(F.label == 9) = -1;
    F.label = MindRoveCode(F.label+7);
    freqs = [0	2	3.90000000000000	5.90000000000000	7.80000000000000	9.80000000000000	11.7000000000000	13.7000000000000	15.6000000000000	17.6000000000000	19.5000000000000	21.5000000000000	23.4000000000000	25.4000000000000	27.3000000000000	29.3000000000000	31.3000000000000	33.2000000000000	35.2000000000000	37.1000000000000	39.1000000000000	41	43	44.9000000000000	46.9000000000000	48.8000000000000	50.8000000000000	52.7000000000000	54.7000000000000	56.6000000000000	58.6000000000000	60.5000000000000	62.5000000000000	64.5000000000000	66.4000000000000	68.4000000000000	70.3000000000000	72.3000000000000	74.2000000000000	76.2000000000000	78.1000000000000	80.1000000000000	82	84	85.9000000000000	87.9000000000000	89.8000000000000	91.8000000000000	93.8000000000000	95.7000000000000	97.7000000000000	99.6000000000000];
    nF = numel(freqs);
    nCh = 8;
    mu = struct('Rest',zeros(nCh,nF), 'Up',zeros(nCh,nF), 'Down', zeros(nCh,nF), 'Left', zeros(nCh, nF), 'Right', zeros(nCh, nF));
    for iCh = 1:nCh
        mu.Rest(iCh,:) = mean(table2array(F(F.label==MindRoveCode.Rest & F.ch==iCh,3:end)),1);
        mu.Up(iCh,:) = mean(table2array(F(F.label==MindRoveCode.UpKeyPress & F.ch==iCh,3:end)),1);
        mu.Down(iCh,:) = mean(table2array(F(F.label==MindRoveCode.DownKeyPress & F.ch==iCh,3:end)),1);
        mu.Left(iCh,:)= mean(table2array(F(F.label==MindRoveCode.LeftKeyPress & F.ch==iCh,3:end)),1);
        mu.Right(iCh,:) = mean(table2array(F(F.label==MindRoveCode.RightKeyPress & F.ch==iCh,3:end)),1);
    end
    F.Properties.UserData = struct('freqs',freqs,'mu',mu);
    F.Properties.UserData.mu = mu;
    F.trial = repelem((1:(size(F,1)/8))',8,1);
    F = movevars(F,'trial','before','label');
end
end