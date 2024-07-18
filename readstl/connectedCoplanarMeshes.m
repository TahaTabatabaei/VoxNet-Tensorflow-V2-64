function coplanarGroups=connectedCoplanarMeshes(vert,showResult)
V=vert.Points;
F=vert.ConnectivityList;


% Compute normals for each face
normals = zeros(size(F, 1), 3);
for i = 1:size(F, 1)
    v1 = V(F(i, 2), :) - V(F(i, 1), :);
    v2 = V(F(i, 3), :) - V(F(i, 1), :);
    n = cross(v1, v2);
    normals(i, :) = n / norm(n);
end

% Identify coplanar faces based on normals and their connectivity
coplanarGroups = {};
faceUsed = false(size(F, 1), 1); % Track if a face is already grouped

tolerance = 1e-3; % Normal vector similarity tolerance

for i = 1:size(F, 1)
    if faceUsed(i)
        continue; % Skip faces already assigned to a group
    end

    % Initialize a new group starting with this face
    currentGroup = i;
    faceUsed(i) = true;
    groupNormals = normals(i, :);

    % Iteratively find and add connected, coplanar faces to the group
    anyAdded = true;
    while anyAdded
        anyAdded = false;
        for j = 1:size(F, 1)
            if faceUsed(j)
                continue;
            end

            % Check if face j is coplanar and connected to any face in the current group
            if norm(cross(normals(j, :), groupNormals)) < tolerance
                % if all(abs(normals(j, :) - groupNormals) < tolerance)
                if isConnected(F(currentGroup,:), F(j,:), V)
                    currentGroup = [currentGroup; j];
                    faceUsed(j) = true;
                    anyAdded = true;
                end
            end
        end
    end

    coplanarGroups{end+1} = currentGroup;
end


coplanarGroups=sortCoplanarGroupsBySurfaceArea(coplanarGroups, F, V);

if showResult>0
    % Visualization
    figure;
    hold on;
    title('Connected Coplanar Meshes');
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    axis equal;
    view(3);

    colors = lines(numel(coplanarGroups));
    bigSurf=0
    for i = 1:numel(coplanarGroups)
        patch('Vertices', V, 'Faces', F(coplanarGroups{i}, :), 'FaceColor', colors(i, :), 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    end
    bigSurf

    hold off;
end
end

function [connected] = isConnected(faces1, faces2, vertices)
% Check if any vertex of faces2 is shared with faces1
verticesList1 = unique(vertices(faces1, :), 'rows');
verticesList2 = unique(vertices(faces2, :), 'rows');
connected = ~isempty(intersect(verticesList1, verticesList2, 'rows'));
end


%#######################################################################################
% Functions
function sortedCoplanarGroups = sortCoplanarGroupsBySurfaceArea(coplanarGroups, F, V)
% Initialize an array to hold the total surface area of each group
groupAreas = zeros(length(coplanarGroups), 1);

% Calculate the surface area for each group
for i = 1:length(coplanarGroups)
    group = coplanarGroups{i};
    totalArea = 0;
    for j = 1:length(group)
        faceIndices = F(group(j), :); % Get vertex indices for the face
        vertices = V(faceIndices, :); % Get the vertices for the face

        % Calculate the area of the triangle and accumulate
        totalArea = totalArea + triangleArea(vertices);
    end
    groupAreas(i) = totalArea;
end

% Sort the groups by total surface area in descending order
[sortedAreas, sortIndices] = sort(groupAreas, 'descend');

% Use the sort indices to reorder the coplanarGroups
sortedCoplanarGroups = coplanarGroups(sortIndices);
end

function area = triangleArea(vertices)
% Calculate the area of a triangle given its vertices
% Heron's formula: A = sqrt(s*(s-a)*(s-b)*(s-c)), where s = (a+b+c)/2
% and a, b, c are the lengths of the sides of the triangle.
a = norm(vertices(2, :) - vertices(1, :));
b = norm(vertices(3, :) - vertices(2, :));
c = norm(vertices(1, :) - vertices(3, :));
s = (a + b + c) / 2;
area = sqrt(s * (s - a) * (s - b) * (s - c));
end
