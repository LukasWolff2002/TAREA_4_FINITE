SetFactory("OpenCASCADE");

// Puntos del cuadrado
Point(1) = {0, 0, 0, 1};
Point(2) = {1, 0, 0, 1};
Point(3) = {1, 1, 0, 1};
Point(4) = {0, 1, 0, 1};

// Líneas
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Superficie
Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Refinamiento hacia (0,0)
Transfinite Curve{1, 2} = 10 Using Progression 1.3;
Transfinite Curve{4, 3} = 10 Using Progression 1/1.3;

// ⚠️ Cambiar orientación de la diagonal
Transfinite Surface {1} = {2, 3, 4, 1};

// Physical Surface
Physical Surface("Dominio") = {1};
