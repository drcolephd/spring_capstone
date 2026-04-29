function parsed_code = parse_mindrove_markers(marker_data)
%PARSE_MINDROVE_MARKERS  Parse marker channel MindRoveCode values
%
% Syntax:
%   parsed_code = parse_mindrove_markers(marker_data);
%
% Inputs:
%   marker_data - Marker-channel data or Decode data logged from mindrove
%                   session. It is already enumerated as `MindRoveCode`
%
% Output:
%   parsed_code - Also MindRoveCode, but the intermediate "Blank" values
%                   are replaced with the most-recent "Non-Blank" value.
%
% See also: read_mindrove, MindRoveCode

arguments
    marker_data MindRoveCode
end

parsed_code = marker_data;
N = numel(parsed_code);
prev_code = MindRoveCode.Blank;
ii = 0;
while ii < N
    ii = ii + 1;
    if parsed_code(ii) == MindRoveCode.Blank
        parsed_code(ii) = prev_code;
    else
        prev_code = marker_data(ii);
    end
end

end