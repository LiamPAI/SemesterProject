cl__1 = 1;
Point(1) = {-0, 0, 0, cl__1};
Point(2) = {0.6, 0, 0, cl__1};
Point(3) = {0.3, 0.3, 0, cl__1};
Point(4) = {-0, 0.6, 0, cl__1};
Point(5) = {-0.3, 0.3, 0, cl__1};
Point(6) = {-0.8, 0.7, 0, cl__1};
Point(7) = {-0.5, 1, 0, cl__1};
Point(8) = {-0.9, 1.6, 0, cl__1};
Point(9) = {-1.2, 1.2, 0, cl__1};
Point(10) = {-1.6, 1.7, 0, cl__1};
Point(11) = {-1.2, 2.2, 0, cl__1};
Point(13) = {-1.9, 3.1, 0, cl__1};
Point(14) = {-1.4, 3.1, 0, cl__1};
Point(15) = {-1.9, 4.5, 0, cl__1};
Point(16) = {-1.4, 4.5, 0, cl__1};
Point(17) = {0.6, 0.6, 0, cl__1};
Point(18) = {0.9, 0.3, 0, cl__1};
Point(19) = {1.3, 0.7, 0, cl__1};
Point(20) = {1, 1, 0, cl__1};
Point(21) = {1.7, 1.5, 0, cl__1};
Point(22) = {1.9, 1.2, 0, cl__1};
Point(23) = {2.8, 1.3, 0, cl__1};
Point(25) = {3.5, 1.7, 0, cl__1};
Point(26) = {3.5, 1.3, 0, cl__1};
Point(27) = {5.2, 1.7, 0, cl__1};
Point(28) = {5.2, 1.3, 0, cl__1};
Point(29) = {-0, -0.6, 0, cl__1};
Point(30) = {-0, -1.4, 0, cl__1};
Point(31) = {0, -2.5, 0, cl__1};
Point(32) = {-0, -3.6, 0, cl__1};
Point(33) = {-0, -4.6, 0, cl__1};
Point(34) = {-0, -5.4, 0, cl__1};
Point(35) = {-0, -6.3, 0, cl__1};
Point(36) = {0.6, -0.6, 0, cl__1};
Point(37) = {0.6, -1.4, 0, cl__1};
Point(38) = {0.6, -2.5, 0, cl__1};
Point(39) = {0.6, -3.6, 0, cl__1};
Point(40) = {0.6, -4.6, 0, cl__1};
Point(41) = {0.6, -5.4, 0, cl__1};
Point(42) = {0.6, -6.3, 0, cl__1};
Point(43) = {2.6, 1.7, 0, cl__1};
Point(44) = {-1.8, 2.5, 0, cl__1};
Line(1) = {1, 3};
Line(2) = {3, 2};
Line(3) = {2, 1};
Line(4) = {15, 16};
Line(5) = {27, 28};
Line(6) = {42, 35};
Spline(7) = {1, 29, 30, 31, 32, 33, 34, 35
};
Spline(8) = {2, 36, 37, 38, 39, 40, 41, 42
};
Spline(9) = {2, 18, 19, 22, 23, 26, 28};
Spline(12) = {3, 4, 7, 8, 11, 14, 16};
Spline(13) = {3, 17, 20, 21, 43, 25, 27};
Spline(14) = {1, 5, 6, 9, 10, 44, 13, 15
};
Curve Loop(1) = {14, 4, -12, -1};
Plane Surface(1) = {1};
Curve Loop(2) = {13, 5, -9, -2};
Plane Surface(2) = {2};
Curve Loop(3) = {7, -6, -8, 3};
Plane Surface(3) = {3};
Curve Loop(4) = {1, 2, 3};
Plane Surface(4) = {4};
Physical Curve("EB1") = {1, 4, 12, 14};
Physical Curve("EB2") = {2, 5, 9, 13};
Physical Curve("EB3") = {3, 6, 7, 8};
Physical Curve("NB1") = {1, 2, 3};
Physical Curve("BC1") = {4};
Physical Curve("BC2") = {5};
Physical Curve("BC3") = {6};
Physical Surface("E1") = {1};
Physical Surface("E2") = {2};
Physical Surface("E3") = {3};
Physical Surface("N1") = {4};
//+
Point(45) = {0.6, -3.9, 0, 1.0};
//+
Delete {
  Point{45}; 
}
//+
Point(45) = {0.038, 0.566, 0, 1.0};
//+
Delete {
  Point{45}; 
}
//+
Point(45) = {-0.318, 0.316, 0, 1.0};
//+
Delete {
  Point{5}; 
}
//+
Delete {
  Point{45}; 
}
//+
Point(45) = {-0.397, 0.9, 0, 1.0};
//+
Point(46) = {-.839, 0.741, 0, 1.0};
//+
Delete {
  Point{46}; Point{45}; 
}
//+
Point(45) = {-1.178, 2.136, 0, 1.0};
//+
Delete {
  Point{11}; 
}
//+
Delete {
  Curve{12}; 
}
//+
Delete {
  Point{45}; 
}
//+
Point(45) = {-1.73, 2.148, 0, 1.0};
//+
Delete {
  Point{45}; 
}
//+
Point(45) = {-0.84, 1.498, 0, 1.0};
//+
Point(46) = {-1.343, 1.358, 0, 1.0};
//+
Delete {
  Point{45}; Point{46}; 
}
//+
Point(45) = {-1.396, 3.055, 0, 1.0};
//+
Point(46) = {-1.885, 2.905, 0, 1.0};
//+
Delete {
  Point{46}; Point{45}; 
}
//+
Point(45) = {2.707, 1.295, 0, 1.0};
//+
Delete {
  Point{45}; 
}
//+
Point(45) = {1.214, 0.62, 0, 1.0};
//+
Point(46) = {0.892, 0.905, 0, 1.0};
//+
Delete {
  Point{20}; Point{19}; 
}
//+
Delete {
  Point{46}; Curve{9}; 
}
//+
Delete {
  Point{45}; 
}
//+
Point(45) = {0.864, 0.264, 0, 1.0};
//+
Point(46) = {0.564, 0.564, 0, 1.0};
//+
Delete {
  Point{46}; Point{45}; 
}
//+
Point(45) = {2.707 , 1.295, 0, 1.0};
//+
Point(46) = {2.511, 1.69, 0, 1.0};
//+
Delete {
  Point{46}; Point{45}; 
}
//+
Point(45) = {1.783, 1.141, 0, 1.0};
//+
Point(46) = {1.579, 1.439, 0, 1.0};
//+
Delete {
  Point{46}; Point{45}; 
}
//+
Point(45) = {3.455, 1.3, 0, 1.0};
//+
Point(46) = {3.432, 1.7, 0, 1.0};
//+
Delete {
  Point{46}; Point{45}; 
}
//+
Point(45) = {0, -1.5, 0, 1.0};
//+
Point(46) = {0.6, -1.5, 0, 1.0};
//+
Delete {
  Point{45}; Point{46}; 
}
//+
Point(45) = {0, -0.625, 0, 1.0};
//+
Point(46) = {0.6, -0.625, 0, 1.0};
//+
Delete {
  Point{45}; Point{46}; 
}
//+
Point(45) = {0, -3.325, 0, 1.0};
//+
Point(46) = {0.6, -3.325, 0, 1.0};
//+
Delete {
  Point{45}; Point{46}; 
}
//+
Point(45) = {0, -4.9, 0, 1.0};
//+
Point(46) = {0.6, -4.9, 0, 1.0};
//+
Delete {
  Point{45}; Point{46}; 
}
//+
Point(45) = {0, -2.4, 0, 1.0};
//+
Point(46) = {00.6, -2.4, 0, 1.0};
//+
Point(47) = {0, -3.325, 0, 1.0};
//+
Delete {
  Point{47}; Point{31}; Point{38}; 
}
//+
Delete {
  Point{45}; Point{46}; 
}
//+
Point(45) = {0, -3.325, 0, 1.0};
//+
Point(46) = {0, -4.175, 0, 1.0};
//+
Delete {
  Point{45}; Point{46}; 
}
