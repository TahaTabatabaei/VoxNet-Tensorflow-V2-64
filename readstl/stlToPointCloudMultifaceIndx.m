function points = stlToPointCloudMultifaceIndx(vert, coplanarGroups, samplesPerUnitLength)

vertices=vert.Points;
faces=vert.ConnectivityList;

% clear vertices; clear faces;
% vertices=[0 0 0;0 0 3;0 2 0; 0 2 3;1 0 0; 1 0 3; 1 2 0; 1 2 3];
% faces=[1 3 5;3 5 7;2 4 6;4 6 8;1 2 5; 2 5 6;3 4 7;4 7 8;1 2 3; 2 3 4;5 6 7;6 7 8];

% Sample points on each coplanar group
points = [];
del=1/samplesPerUnitLength;
for g=1:size(coplanarGroups,2)
    v1 = vertices(faces(coplanarGroups{g}(1),1),:);
    v2 = vertices(faces(coplanarGroups{g}(1),2),:);
    v3 = vertices(faces(coplanarGroups{g}(1),3),:);
    [T, R] = plane_to_xy ([v1;v2;v3]);
    Qt=[];
    for i = 1:size(coplanarGroups{g},1)
        v1 = vertices(faces(coplanarGroups{g}(i),1),:);
        v2 = vertices(faces(coplanarGroups{g}(i),2),:);
        v3 = vertices(faces(coplanarGroups{g}(i),3),:);
        tri=[v1;v2;v3];
        Q=(R*(tri'+T))';
        Qt=[Qt; Q];
    end
    del=1/samplesPerUnitLength;
    [X,Y] = meshgrid(min(Qt(:,1)):del:max(Qt(:,1)),min(Qt(:,2)):del:max(Qt(:,2)) );
    X = X(:);
    Y = Y(:);
    Z=zeros(size(X));
    grids = [X Y Z];
    gPoints=[];
    indF=[];
    for i = 1:size(coplanarGroups{g},1)
        ii=(i-1)*3+1;
        points1=inTriangle(Qt(ii:ii+2,:), grids);
        gPoints=[gPoints; points1];
        indF=[indF; coplanarGroups{g}(i)*ones(size(points1,1),1)];
        % [~, indices] = ismember(grids, points1);
        % grids(indices(indices > 0),:) = [];
    end

    if size(gPoints,1)>0
        invT=-T;
        invR=inv(R);
        gPoints=((invR * gPoints') + invT)';
    end

    points=[points;[gPoints, indF]];

end
end


%####################################################################################
function points = sampleTriangle1(tri,n)
% This function takes n sample points evenly on a triangle
% and returns a matrix of points with x, y, and z coordinates

% Find the coefficients of the plane equation
coeffPlane = cross(tri(3,:)-tri(1,:), tri(2,:)-tri(1,:));
offset = dot(coeffPlane, tri(3,:));

% Create a meshgrid of x and y values
[X,Y] = meshgrid(linspace(min(tri(:,1)),max(tri(:,1)),n));
X = X(:);
Y = Y(:);

% Solve for z values using the plane equation
if coeffPlane(3) ~= 0
    Z = (offset - coeffPlane(1) * X - coeffPlane(2) * Y) / coeffPlane(3);
else
    % If coeffPlane(3) is 0, set Z to the Z-coordinate of any triangle vertex
    Z = repmat(tri(1,3), size(X)); % Example: using Z-coordinate of the first vertex
end

% Combine x, y, and z values into a matrix
points = [X Y Z];

% Convert the cartesian coordinates to barycentric coordinates
triAng=triangulation([1,2,3], tri);
baryCoord = cartesianToBarycentric(triAng, ones(size(X)), points);

% Filter out the points that are not inside the triangle
TFinTri = all(baryCoord>=0 & baryCoord<=1,2);
points(~TFinTri,:) = [];

% Plot the points on the triangle
scatter3(points(:, 1), points(:, 2), points(:,3), '.')
end
