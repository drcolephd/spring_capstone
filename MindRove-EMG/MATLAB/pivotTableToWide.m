function [W, Feature, Label, freqs, ch] = pivotTableToWide(F)
% Pivots a table from tall format to wide format.
%
% Parameters:
% - F: a table with columns 'trial', 'label', 'ch', and features ('feature_1', 'feature_2', ...)
%
% Returns:
% - W: "Wide" table
% - Feature: Feature data
% - Label: Labels corresponding to rows of returned features

nFreq = numel(F.Properties.UserData.freqs);
data = table2array(F(:,4:end))';
Feature = reshape(data, 8*nFreq, [])';
Label = F.label(1:8:end);
freqs = repmat(F.Properties.UserData.freqs,1,8);
ch = repelem(1:8,1,nFreq);

vname = cell(numel(ch)+1,1);
vname{1} = 'Label';
for ii = 1:numel(ch)
    vname{ii+1} = sprintf('Feature_%dHz_Ch%d',round(freqs(ii)),ch(ii));
end
Feature = mat2cell(Feature,size(Feature,1),ones(1,size(Feature,2)));
W = table(Label,Feature{:});
W.Properties.VariableNames = vname;
end
