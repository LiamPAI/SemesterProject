Point(1) = {-1, 0, 0, 1.0};
//+
Point(2) = {1, 0, 0, 1.0};
//+
Point(3) = {0, 1, 0, 1.0};
//+
Point(4) = {0, -1, 0, 1.0};
//+
Point(5) = {-3, 4, 0, 1.0};
//+
Point(6) = {-3, -4, 0, 1.0};
//+
Point(7) = {-4, -3, 0, 1.0};
//+
Point(8) = {-4, 3, 0, 1.0};
//+
Point(9) = {4, 3, 0, 1.0};
//+
Point(10) = {4, -3, 0, 1.0};
//+
Point(11) = {3, -4, 0, 1.0};
//+
Point(12) = {3, 4, 0, 1.0};
//+
Line(1) = {1, 3};
//+
Line(2) = {3, 2};
//+
Line(3) = {2, 4};
//+
Line(4) = {4, 1};
//+
Line(5) = {8, 5};
//+
Line(6) = {12, 9};
//+
Line(7) = {10, 11};
//+
Line(8) = {6, 7};
//+
Spline(9) = {4, 6};
//+
Spline(10) = {1, 7};
//+
Spline(11) = {1, 8};
//+
Spline(12) = {3, 5};
//+
Spline(13) = {3, 12};
//+
Spline(14) = {2, 9};
//+
Spline(15) = {2, 10};
//+
Spline(16) = {4, 11};
//+
Physical Curve("NB1", 17) = {1, 2, 3, 4};
//+
Physical Curve("EB1", 18) = {10, 4, 9, 8};
//+
Physical Curve("EB2", 19) = {1, 11, 5, 12};
//+
Physical Curve("EB3", 20) = {2, 13, 6, 14};
//+
Physical Curve("EB4", 21) = {3, 15, 7, 16};
//+
Curve Loop(1) = {10, -8, -9, 4};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {1, 12, -5, -11};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {2, 14, -6, -13};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {3, 16, -7, -15};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {4, 1, 2, 3};
//+
Plane Surface(5) = {5};
//+
Physical Surface("N1", 22) = {5};
//+
Physical Surface("E1", 23) = {1};
//+
Physical Surface("E2", 24) = {2};
//+
Physical Surface("E3", 25) = {3};
//+
Physical Surface("E4", 26) = {4};
