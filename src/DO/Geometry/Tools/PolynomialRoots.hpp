
namespace DO {

  void computeRoots(bool& hasRealSolutions,
                    std::complex<double>& x1,
                    std::complex<double>& x2,
                    double a, double b, double c)
  {
    double delta = b*b-4*a*c;
    x1 = (-b - sqrt(complex<double>(delta))) / (2*a);
    x2 = (-b + sqrt(complex<double>(delta))) / (2*a);
    if(delta >= 0)
      hasRealSolutions = true;
    else
      hasRealSolutions = false;
  }

  // Discriminant precision: 1e-3.
  void computeRoots(std::complex<double>& z1, std::complex<double>& z2,
                    std::complex<double>& z3,
                    double a, double b, double c, double d)
  {
    const double pi = 3.14159265358979323846;

    b /= a;
    c /= a;
    d /= a;
    a = 1.0;

    // Cardano's formula.
    const double p = (3*c-b*b)/3;
    const double q = (-9*c*b + 27*d + 2*b*b*b)/27;
    const double delta = q*q + 4*p*p*p/27;

    if(delta < -1e-3)
    {
      const double theta = acos( -q/2*sqrt(27/(-p*p*p)) )/3.0;
      z1 = 2*sqrt(-p/3)*cos( theta );
      z2 = 2*sqrt(-p/3)*cos( theta + 2*pi/3);
      z3 = 2*sqrt(-p/3)*cos( theta + 4*pi/3);
    }
    else if(delta <= 1e-3)
    {
      z1 = 3*q/p;
      z2 = z3 = -3*q/(2*p);
    }
    else
    {
      double r1 = (-q+sqrt(delta))/2.0;
      double r2 = (-q-sqrt(delta))/2.0;
      double u = r1 < 0 ? -pow(-r1, 1.0/3.0) : pow(r1, 1.0/3.0);
      double v = r2 < 0 ? -pow(-r2, 1.0/3.0) : pow(r2, 1.0/3.0);
      complex<double> j(-0.5, sqrt(3.0)*0.5);
      z1 = u + v;
      z2 = j*u+conj(j)*v;
      z3 = j*j*u+conj(j*j)*v;
    }

    z1 -= b/(3*a);
    z2 -= b/(3*a);
    z3 -= b/(3*a);
  }

  // Involves the precision of the cubic equation solver: (1e-3.)
  void computeRoots(std::complex<double>& z1, std::complex<double>& z2,
                    std::complex<double>& z3, std::complex<double>& z4,
                    double a4, double a3, double a2, double a1, double a0)
  {
    a3 /= a4; a2/= a4; a1 /= a4; a0 /= a4; a4 = 1.0;

    double coeff[4];
    coeff[3] = 1.0;
    coeff[2] = -a2;
    coeff[1] = a1*a3 - 4.0*a0;
    coeff[0] = 4.0*a2*a0 - a1*a1 - a3*a3*a0;

    complex<double> y1, y2, y3;
    /*cout << "Intermediate cubic polynomial" << endl;
    printPolynomial(coeff, 3);*/
    solveCubicEquation(y1, y2, y3, coeff[3], coeff[2], coeff[1], coeff[0]);

    double yr = real(y1);
    double yi = fabs(imag(y1));
    if(yi > fabs(imag(y2)))
    {
      yr = real(y2);
      yi = fabs(imag(y2));
    }
    if(yi > fabs(imag(y3)))
    {
      yr = real(y3);
      yi = fabs(imag(y3));
    }

    complex<double> radicand = a3*a3/4.0 - a2 + yr;
    complex<double> R( sqrt(radicand) );
    complex<double> D, E;

    if(abs(R) > 1e-3)
    {
      D = sqrt( 3.0*a3*a3/4.0 - R*R - 2.0*a2 + (4.0*a3*a2 - 8.0*a1 - a3*a3*a3)/(4.0*R) );
      E = sqrt( 3.0*a3*a3/4.0 - R*R - 2.0*a2 - (4.0*a3*a2 - 8.0*a1 - a3*a3*a3)/(4.0*R) );
    }
    else
    {
      D = sqrt( 3.0*a3*a3/4.0 - 2.0*a2 + 2.0*sqrt(yr*yr - 4.0*a0) );
      E = sqrt( 3.0*a3*a3/4.0 - 2.0*a2 - 2.0*sqrt(yr*yr - 4.0*a0) );
    }

    z1 =  R/2.0 + D/2.0;
    z2 =  R/2.0 - D/2.0;
    z3 = -R/2.0 + E/2.0;
    z4 = -R/2.0 - E/2.0;

    // Check Viete's formula.
    /*double p = a2 - 3*a3*a3/8;
    double q = a1 - a2*a3/2 + a3*a3*a3/8;
    double r = a0 - a1*a3/4 + a2*a3*a3/16 - 3*a3*a3*a3*a3/256;

    cout << "-2p = " << -2*p << endl;
    cout << pow(z1,2) + pow(z2,2) + pow(z3,2) + pow(z4,2) << endl;
    cout << "-3*q = " << -3*q << endl;
    cout << pow(z1,3) + pow(z2,3) + pow(z3,3) + pow(z4,3) << endl;
    cout << "2p^2 - 4r = " << 2*p*p - 4*r << endl;
    cout << pow(z1,4) + pow(z2,4) + pow(z3,4) + pow(z4,4) << endl;
    cout << "5pq = " << 5*p*q << endl;
    cout << pow(z1,5) + pow(z2,5) + pow(z3,5) + pow(z4,5) << endl;*/

    z1 -= a3/4;
    z2 -= a3/4;
    z3 -= a3/4;
    z4 -= a3/4;
  }

  void checkQuadraticEquationSolver()
  {
    // check quadratic equation solver
    bool hasRealSolutions;
    complex<double> x1, x2;
    double p[3] ={1.0, 0.0, 2.0};
    printPolynomial(p, 2);
    solveQuadraticEquation(hasRealSolutions, x1, x2, p[2], p[1], p[0]);
    cout << "x1 = " << x1 << " and x2 = " << x2 << endl;
    cout << "P(" << x1 << ") = " << computePolynomial(x1, p, 2) << endl;
    cout << "P(" << x2 << ") = " << computePolynomial(x2, p, 2) << endl;
    cout << endl;
  }

  void checkCubicEquationSolver()
  {
    // check quadratic equation solver
    complex<double> x1, x2, x3;
    for(int i = 0; i < 10; ++i)
    {
      cout << "iteration " << i << endl;
      double p[4] ={
        static_cast<double>(-rand()%10),
        static_cast<double>(-rand()%10),
        static_cast<double>(-rand()%10),
        static_cast<double>(-rand()%10)
      };
      printPolynomial(p, 3);

      solveCubicEquation(x1, x2, x3, p[3], p[2], p[1], p[0]);
      cout << "x1 = " << x1 << " and x2 = " << x2 << " and x3 = " << x3 << endl;
      cout << "|P(" << x1 << ")| = " << abs(computePolynomial(x1, p, 3)) << endl;
      cout << "|P(" << x2 << ")| = " << abs(computePolynomial(x2, p, 3)) << endl;
      cout << "|P(" << x3 << ")| = " << abs(computePolynomial(x3, p, 3)) << endl;
      cout << endl;
    }
  }

  void checkQuarticEquationSolver()
  {
    // check quadratic equation solver
    complex<double> x1, x2, x3, x4;
    for(int i = 0; i < 10; ++i)
    {
      cout << "iteration " << i << endl;
      double p[5] ={
        static_cast<double>(rand()%100000),
        static_cast<double>(-rand()%10),
        static_cast<double>(-rand()%10),
        static_cast<double>(-rand()%10),
        static_cast<double>(-rand()%10)+1
      };
      printPolynomial(p, 4);
      solveQuarticEquation(x1, x2, x3, x4,
                           p[4], p[3], p[2], p[1], p[0]);

      cout << "x1 = " << x1 << " and x2 = " << x2 << endl;
      cout << "x3 = " << x3 << " and x4 = " << x4 << endl;
      cout << "|P(" << x1 << ")| = " << abs(computePolynomial(x1, p, 4)) << endl;
      cout << "|P(" << x2 << ")| = " << abs(computePolynomial(x2, p, 4)) << endl;
      cout << "|P(" << x3 << ")| = " << abs(computePolynomial(x3, p, 4)) << endl;
      cout << "|P(" << x4 << ")| = " << abs(computePolynomial(x4, p, 4)) << endl;
      cout << endl;
    }
  }

} /* namespace DO */
