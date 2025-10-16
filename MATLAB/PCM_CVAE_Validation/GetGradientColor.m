function RGBColor = GetGradientColor(row, col)
% GetGradientColor - Generate an RGB gradient color from red to blue
%
% Description:
%   This function produces an RGB color value corresponding to a specific
%   position in a red-to-blue gradient table. The gradient transitions from
%   red (top row) to blue (bottom row), while the brightness increases from
%   left to right (toward white).
%
% Inputs:
%   row - Current row index (1–10), controls the main hue transition (red → blue)
%   col - Current column index (1–10), controls the brightness level
%
% Output:
%   RGBColor - A structure containing normalized RGB components in the range [0, 1]:
%       .R - Red component
%       .G - Green component
%       .B - Blue component
%
% Example:
%   color = GetGradientColor(3, 8);
%   plot(1,1,'o','MarkerFaceColor',[color.R color.G color.B],'MarkerEdgeColor','none');

% -------------------------------------------------------------------------

% Initialize a 10×12×3 matrix for RGB values
img = zeros(10, 12, 3);

% Define row-wise main hue: interpolate from red (255,0,0) to blue (0,0,255)
for i = 1:10
    t_row = (i - 1) / 9; % Normalized factor from 0 to 1
    main_color = (1 - t_row) * [255, 0, 0] + t_row * [0, 0, 255]; % Linear interpolation (red → blue)

    % Within each row, gradually lighten the color toward white
    for j = 1:12
        t_col = (j - 1) / 11; % Normalized factor from 0 to 1
        light_color = (1 - t_col) * main_color + t_col * [255, 255, 255];
        img(i, j, :) = light_color / 255; % Normalize to [0, 1]
    end
end

% Trim to 10 columns (remove last two columns)
img(:, 11:12, :) = [];

% Retrieve the RGB values for the specified (row, col)
RGBColor.R = img(row, col, 1);
RGBColor.G = img(row, col, 2);
RGBColor.B = img(row, col, 3);

end