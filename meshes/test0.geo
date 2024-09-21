// Gmsh project created on Tue Apr 30 11:02:36 2024
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0, 1, 0, 1.0};
//+
Point(3) = {2, 1, 0, 1.0};
//+
Point(4) = {2, 0.5, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
Physical Curve(5) = {1};
//+
Physical Curve(6) = {2};
//+
Physical Curve(7) = {4, 3};
//+
Physical Surface(8) = {1};
//+
Physical Curve("Traction1", 9) = {2};
//+
Physical Curve("Traction1", 9) += {2};
//+
Physical Curve(6) -= {2};
//+
Physical Curve(5) -= {4, 3};
//+
Physical Curve(7) -= {4, 3};
//+
Physical Curve(5) -= {1};
//+
Physical Curve("Distance0", 10) = {1};
//+
Physical Curve("Traction0", 11) = {4, 3};
