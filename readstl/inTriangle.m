function points=inTriangle(tri, grids)
X=grids(:,1);
Y=grids(:,2);
Q=[tri;tri(1,:)];
[in,on]=inpolygon(X,Y,Q(:,1),Q(:,2));inOn=in|on;
PX=X(inOn); PY=Y(inOn);
points=[PX,PY,zeros(size(PX,1),1)];
end
