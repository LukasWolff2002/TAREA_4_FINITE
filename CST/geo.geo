SetFactory("OpenCASCADE");

// Puntos
Point(1) = {0, 0, 0, 1};
Point(2) = {1, 0, 0, 1};
Point(3) = {1, 1, 0, 1};
Point(4) = {0, 1, 0, 1};

// Líneas (sentido antihorario)
Line(1) = {1, 2};  // inferior
Line(2) = {2, 3};  // derecho
Line(3) = {3, 4};  // superior
Line(4) = {4, 1};  // izquierdo

// Superficie
Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Parámetros de mallado
n = 59;
r = 1.1;

// Refinamiento progresivo
Transfinite Curve{1, 2} = 59 Using Progression 1.1;
Transfinite Curve{3, 4} = n Using Progression 1/r;

// ⚠️ Cambia orientación de malla, pero NO las líneas físicas
Transfinite Surface {1} = {2, 3, 4, 1};

// Physical tags
Physical Line("Dirichlet 1") = {1};  // borde inferior
Physical Line("Dirichlet 2") = {2};  // borde derecho
Physical Line("Dirichlet 3") = {3};  // borde superior
Physical Line("Dirichlet 4") = {4};  // borde izquierdo

Physical Surface("Dominio") = {1};
