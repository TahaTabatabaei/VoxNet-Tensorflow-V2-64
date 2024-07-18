samplesPerUnitLength=3;
stlFile='NGen_AI';

vert = stlread([stlFile,'.stl']);
coplanarGroups=connectedCoplanarMeshes(vert,1);

points = stlToPointCloudMultifaceIndx(vert, coplanarGroups, samplesPerUnitLength);
modelPoints=points(:,1:3);

PtCloud=pointCloud(modelPoints);

save([stlFile,'_Indx',num2str(samplesPerUnitLength),'.mat'], "points", "coplanarGroups");
pcwrite(PtCloud,[stlFile,'_stp',num2str(samplesPerUnitLength),'.ply'])




%###########################################################################
function distMat = euclideanDistanceTwoPointClouds(scannerPts,modelPts)
% calculate the euclidean distance for every point in sample point cloud to
% the closest point in the reference point cloud (number of columns must
% match)
% INPUT:
%     scannerPts = M x 3 matrix
%     modelPts    = P x 3 matrix
% OUTPUT:
%     distMat   = P x 1 matrix

[Idx, distMat]=knnsearch(scannerPts,modelPts);

end

