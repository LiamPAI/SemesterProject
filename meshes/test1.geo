// Gmsh project created on Wed May  8 11:47:47 2024
//+
Point(1) = {0.0, 0.0, 0.0, 0.25};
//+
Point(2) = {1, 0.0, 0.0, 0.25};
//+
Recursive Delete {
  Point{2}; 
}
//+
Point(2) = {1.0, 0.0, 0.0, 0.25};
//+
Point(3) = {0.0, 2.0, 0.0, 0.25};
//+
Point(4) = {2.0, 2.0, 0.0, 0.25};
//+
Point(5) = {2.0, 1.0, 0.0, 0.25};
//+
Point(6) = {2.0, 0.0, 0.0, 0.25};
//+
Circle(1) = {2, 6, 5};
//+
Recursive Delete {
  Curve{1}; Point{5}; Point{2}; 
}
//+
Point(5) = {0.5, 0.0, 0.0, 0.25};
//+
Point(6) = {2.0, 1.5, 0.0, 0.25};
//+
Point(7) = {2.0, 0.0, 0.0, 0.25};
//+
Circle(1) = {5, 7, 6};
//+
Line(2) = {1, 5};
//+
Line(3) = {1, 3};
//+
Line(4) = {3, 4};
//+
Line(5) = {4, 6};
//+
Curve Loop(1) = {4, 5, -1, -2, 3};
//+
Plane Surface(1) = {1};
//+
Physical Curve("Distance0", 6) = {3};
//+
Physical Curve("Traction1", 7) = {4};
//+
Physical Curve("Traction0", 8) = {5, 1, 2};
//+
Physical Surface("Surface", 9) = {1};
