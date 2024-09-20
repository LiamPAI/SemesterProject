// Gmsh project created on Mon Sep  9 10:09:29 2024
//+
Point(1) = {0, 1, 0, 1.0};
//+
Point(2) = {-1, 0, 0, 1.0};
//+
Point(3) = {1, 0, 0, 1.0};
//+
Point(4) = {-4, 3, 0, 1.0};
//+
Point(5) = {-3, 4, 0, 1.0};
//+
Line(1) = {2, 1};
//+
Line(2) = {1, 3};
//+
Line(3) = {3, 2};
//+
Curve Loop(1) = {1, 2, 3};
//+
Plane Surface(1) = {1};
//+
Physical Surface("N1", 4) = {1};
//+
Physical Curve("NB1", 5) = {1, 2, 3};
//+
Point(6) = {-3.5, 7, 0, 1.0};
//+
Point(7) = {-4.91, 7, 0, 1.0};
//+
Line(4) = {7, 6};
//+
Spline(5) = {2, 4, 7};
//+
Spline(6) = {1, 5, 6};
//+
Curve Loop(2) = {5, 4, -6, -1};
//+
Plane Surface(2) = {2};
//+
Physical Curve("EB1", 7) = {1, 5, 4, 6};
//+
Physical Surface("E1", 8) = {2};
//+
Point(8) = {5, 6, 0, 1.0};
//+
Point(9) = {6, 5, 0, 1.0};
//+
Spline(7) = {1, 8};
//+
Spline(8) = {9, 3};
//+
Line(9) = {8, 9};
//+
Curve Loop(3) = {7, 9, 8, -2};
//+
Plane Surface(3) = {3};
//+
Physical Curve("EB2", 10) = {7, 9, 8, 2};
//+
Physical Surface("E2", 11) = {3};
//+
Point(10) = {-3, -4, 0, 1.0};
//+
Point(11) = {-1, -4, 0, 1.0};
//+
Point(12) = {-2, -7, 0, 1.0};
//+
Point(13) = {0, -7, 0, 1.0};
//+
Point(14) = {-1, -8.73, 0, 1.0};
//+
Line(10) = {12, 13};
//+
Line(11) = {13, 14};
//+
Line(12) = {14, 12};
//+
Curve Loop(4) = {10, 11, 12};
//+
Plane Surface(4) = {4};
//+
Physical Curve("NB2", 13) = {10, 11, 12};
//+
Physical Surface("N2", 14) = {4};
//+
Spline(13) = {2, 10, 12};
//+
Spline(14) = {13, 11, 3};
//+
Curve Loop(5) = {13, 10, 14, 3};
//+
Plane Surface(5) = {5};
//+
Physical Curve("EB3", 15) = {13, 3, 14, 10};
//+
Physical Surface("E3", 16) = {5};
//+
Point(15) = {-6, -9.5, 0, 1.0};
//+
Point(16) = {-6, -7.5, 0, 1.0};
//+
Point(17) = {-10, -8.5, 0, 1.0};
//+
Point(18) = {-10, -6.5, 0, 1.0};
//+
Point(19) = {6, -9, 0, 1.0};
//+
Point(20) = {6, -7, 0, 1.0};
//+
Line(15) = {17, 18};
//+
Line(16) = {20, 19};
//+
Spline(17) = {12, 16, 18};
//+
Spline(18) = {17, 15, 14};
//+
Spline(19) = {14, 19};
//+
Spline(20) = {20, 13};
//+
Curve Loop(6) = {15, -17, -12, -18};
//+
Plane Surface(6) = {6};
//+
Curve Loop(7) = {11, 19, -16, 20};
//+
Plane Surface(7) = {7};
//+
Physical Curve("EB4", 21) = {12, 18, 15, 17};
//+
Physical Curve("EB5", 22) = {11, 20, 16, 19};
//+
Physical Surface("E4", 23) = {6};
//+
Physical Surface("E5", 24) = {7};
