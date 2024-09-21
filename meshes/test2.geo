// Gmsh project created on Mon May  6 11:21:20 2024
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 0.25};
//+
Point(2) = {1, 0, 0, 0.25};
//+
Point(3) = {0, 2, 0, 0.25};
//+
Point(4) = {1, 2, 0, 0.25};
//+
Point(5) = {2, 2, 0, 0.25};
//+
Point(6) = {2, 3, 0, 0.25};
//+
Point(7) = {2, 4, 0, 0.25};
//+
Point(8) = {4, 4, 0, 0.25};
//+
Point(9) = {4, 3, 0, 0.25};
//+
Line(1) = {1, 3};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 4};
//+
Line(4) = {6, 9};
//+
Line(5) = {9, 8};
//+
Line(6) = {8, 7};
//+
Circle(7) = {4, 5, 6};
//+
Circle(8) = {3, 5, 7};
//+
Physical Curve(9) = {6, 8, 4, 5, 7, 3, 2, 1};
//+
Curve Loop(1) = {6, -8, -1, 2, 3, 7, 4, 5};
//+
Plane Surface(1) = {1};
//+
Physical Surface(10) = {1};
