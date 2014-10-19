
TEST(DO_Geometry_Test, csgTest)
{
  Ellipse E(180, 100, 0., TestParams::center);
  AffineCone2 K(Point2d(1,0), Point2d(0,1), TestParams::center);

  CSG::Singleton<Ellipse> ell(E);
  CSG::Singleton<AffineCone2> cone(K);
  auto inter = ell*cone;

  int estimatedArea = sweepCountPixels(
    [&](const Point2d& p){ return inter.contains(p); },
    TestParams::debug );

  double trueArea = area(E)/4.;
  double relativeError = fabs(estimatedArea -trueArea)/trueArea;
  EXPECT_NEAR(relativeError, 0, 1e-1);
}

TEST(DO_Geometry_Test, ellipseSectorArea)
{
  Ellipse E(250, 150, toRadian(75.), TestParams::center);

  CSG::Singleton<Ellipse> ell(E);
  Image<Rgb8> buffer(TestParams::w, TestParams::h);

  if (TestParams::debug)
  {
    if (!getActiveWindow())
      setAntialiasing(openWindow(TestParams::w, TestParams::h));
    clearWindow();
  }

  try {

    int steps = 18;
    for (int i0 = 0; i0 <= steps; ++i0)
    {
      double theta0 = i0*2*M_PI/steps;
      for (int i1 = i0+1; i1 <= i0+steps; ++i1)
      {
        double theta1 = i1*2*M_PI/steps;

        // Compute the sector area in a closed form.
        double analyticSectorArea = sectorArea(E, theta0, theta1);

        // Build constructive solid geometry.
        CSG::Singleton<AffineCone2> cone (
          affineCone2(theta0+E.orientation(),
          theta1+E.orientation(),
          E.center()) );
        auto E_and_Cone = ell*cone;
        auto E_minus_Cone = ell - E_and_Cone;

        // Use the CSGs to estimate the sector area by pixel counting.
        auto inside_E_and_Cone = [&](const Point2d& p){ 
          return E_and_Cone.contains(p);
        };
        auto inside_E_minus_Cone = [&](const Point2d& p){ 
          return E_minus_Cone.contains(p);
        };

        // Use the lambda functors to estimate the elliptic sector area.
        int estimatedSectorArea;
        if (i1 - i0 < steps/2)
          estimatedSectorArea = sweepCountPixels(inside_E_and_Cone, buffer);
        else if (abs(i1-i0) == steps/2)
          estimatedSectorArea = area(E) / 2.;
        else
          estimatedSectorArea = sweepCountPixels(inside_E_minus_Cone, buffer);

        // Absolute error and relative error.
        double absError = fabs(estimatedSectorArea -analyticSectorArea);
        double relError = absError/estimatedSectorArea;

        if (TestParams::debug)
          display(buffer);

        double thres = 1e-1;
        EXPECT_NEAR(relError, 0, thres);
        if (TestParams::debug && relError > thres)
        {
          printStage("Numerical error");
          cout << i0 << "     " << i1 << endl;
          CHECK(abs(i1-i0));
          CHECK(analyticSectorArea);
          CHECK(estimatedSectorArea);
          CHECK(absError);
          CHECK(relError);
          getKey();
        }
      }
    }
  }
  catch (exception& e)
  {
    cout << e.what() << endl;
    getKey();
  }
}
