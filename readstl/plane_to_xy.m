%************************************************************
function [T, R] = plane_to_xy (tri)
% Input: three points tri that define a plane
% Output: translation vector T and final rotation matrix R

% Define the three points as column vectors
P1 = tri(1,:)';
P2 = tri(2,:)';
P3 = tri(3,:)';

% Find the translation vector
T = -P1;

% Find the first rotation matrix
R1 = vrrotvec2mat (vrrotvec (P2 - P1, [1 0 0]));

% Apply the translation and the first rotation to the three points
Q1 = R1 * (P1 + T);
Q2 = R1 * (P2 + T);
Q3 = R1 * (P3 + T);
n=cross(Q2,Q3);

% Find the second rotation matrix
R2 = vrrotvec2mat (vrrotvec (n, [0 0 1]));

% Find the final rotation matrix
R = R2 * R1;

end