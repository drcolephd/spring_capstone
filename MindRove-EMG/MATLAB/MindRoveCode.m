classdef MindRoveCode < int32
    % MindRoveCode Enumeration of event codes for MindRove system events.
    %
    % This enumeration class defines a set of integer codes representing
    % various events within the MindRove system, including directional 
    % prompts, game state changes, user actions, and real-time plot markers. 
    % These codes provide a standardized way to track and respond to events 
    % programmatically, facilitating system interaction and data logging.
    %
    % Enumeration Members:
    %   Default
    %       Blank                       - (0) Default marker value
    %
    %   Arrow Directions:
    %       UpArrowShown                - (2) Up arrow prompt displayed
    %       RightArrowShown             - (3) Right arrow prompt displayed
    %       DownArrowShown              - (4) Down arrow prompt displayed
    %       LeftArrowShown              - (5) Left arrow prompt displayed
    %
    %   Direction Game Events:
    %       Rest                        - (6) Direction game opened
    %       UpKeyPress                  - (7) Up key pressed by user
    %       RightKeyPress               - (8) Right key pressed by user
    %       DownKeyPress                - (9) Down key pressed by user
    %       LeftKeyPress                - (10) Left key pressed by user
    %       DirectionGameStart          - (11) Direction game started
    %       DirectionGameEnd            - (12) Direction game ended
    %
    %   Pac-Man Game Events:
    %       PacManGameOpen              - (13) Pac-Man game opened
    %       PacManLevelStart            - (14) Pac-Man level started
    %       PacManPelletConsumed        - (15) Pellet consumed in Pac-Man
    %       PacManDied                  - (16) Pac-Man character died
    %       PacManConsumedPowerup       - (17) Powerup consumed in Pac-Man
    %       PacManConsumedGhost         - (18) Ghost consumed in Pac-Man
    %       PacManLevelClear            - (19) Level cleared in Pac-Man
    %       PacManLevelUp               - (20) Pac-Man leveled up
    %       PacManLevelStop             - (21) Pac-Man level stopped
    %
    %   Real-Time Plot Markers:
    %       RealTimePlotMarker0         - (30) Plot marker 0 in real-time plot
    %       RealTimePlotMarker1         - (31) Plot marker 1 in real-time plot
    %       RealTimePlotMarker2         - (32) Plot marker 2 in real-time plot
    %       RealTimePlotMarker3         - (33) Plot marker 3 in real-time plot
    %       RealTimePlotMarker4         - (34) Plot marker 4 in real-time plot
    %       RealTimePlotMarker5         - (35) Plot marker 5 in real-time plot
    %       RealTimePlotMarker6         - (36) Plot marker 6 in real-time plot
    %       RealTimePlotMarker7         - (37) Plot marker 7 in real-time plot
    %       RealTimePlotMarker8         - (38) Plot marker 8 in real-time plot
    %       RealTimePlotMarker9         - (39) Plot marker 9 in real-time plot
    %
    %   Real-Time Calibration Events:
    %       RealTimeRestCalibrationClick    - (40) Rest calibration clicked
    %       RealTimeRestCalibrationStart    - (41) Rest calibration started
    %       RealTimeRestCalibrationEnd      - (42) Rest calibration ended
    %       RealTimeThresholdCalibrationClick - (43) Threshold calibration clicked
    %       RealTimeThresholdCalibrationStart - (44) Threshold calibration started
    %       RealTimeThresholdCalibrationEnd - (45) Threshold calibration ended
    %
    % Example:
    %   % Using an enumeration member
    %   if eventCode == MindRoveCode.UpArrowShown
    %       disp('Up arrow prompt is shown to the user.');
    %   end
    %
    % See also: int32, enumeration
    enumeration
        % Default
        Blank (0)
        Click (1)

        % Arrow Directions
        UpArrowShown (2)
        RightArrowShown (3)
        DownArrowShown (4)
        LeftArrowShown (5)
        
        % Direction Game Events
        Rest (6)
        UpKeyPress (7)
        RightKeyPress (8)
        DownKeyPress (9)
        LeftKeyPress (10)
        DirectionGameStart (11)
        DirectionGameEnd (12)
        
        % Pac-Man Game Events
        PacManGameOpen (13)
        PacManLevelStart (14)
        PacManPelletConsumed (15)
        PacManDied (16)
        PacManConsumedPowerup (17)
        PacManConsumedGhost (18)
        PacManLevelClear (19)
        PacManLevelUp (20)
        PacManLevelStop (21)
        
        % Real-Time Plot Markers
        RealTimePlotMarker0 (30)
        RealTimePlotMarker1 (31)
        RealTimePlotMarker2 (32)
        RealTimePlotMarker3 (33)
        RealTimePlotMarker4 (34)
        RealTimePlotMarker5 (35)
        RealTimePlotMarker6 (36)
        RealTimePlotMarker7 (37)
        RealTimePlotMarker8 (38)
        RealTimePlotMarker9 (39)
        
        % Real-Time Calibration Events
        RealTimeRestCalibrationClick (40)
        RealTimeRestCalibrationStart (41)
        RealTimeRestCalibrationEnd (42)
        RealTimeThresholdCalibrationClick (43)
        RealTimeThresholdCalibrationStart (44)
        RealTimeThresholdCalibrationEnd (45)
    end
end
