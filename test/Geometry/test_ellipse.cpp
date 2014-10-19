TEST(DO_Geometry_Test, ellipseSegmentArea)
{
  Ellipse E(270, 150, toRadian(42.), TestParams::center);
  Image<Rgb8> buffer(TestParams::w, TestParams::h);

  if (TestParams::debug)
  {
    if (!getActiveWindow())
      setAntialiasing(openWindow(TestParams::w, TestParams::h));
    clearWindow();
  }

  try
  {
    int steps = 18;

    for (int i0 = 0; i0 <= steps; ++i0)
    {
      double theta0 = i0*2*M_PI/steps;
      for (int i1 = i0+1; i1 < i0+steps; ++i1)
      {
        double theta1 = i1*2*M_PI/steps;

        const Point2d a(E(theta0));
        const Point2d b(E(theta1));
        const Point2d& o = E.center();
        Triangle t(o, a, b);

        auto insideSegment = [&](const Point2d& p) -> bool
        { return (ccw(a,b,p) == -1) && inside(p, E); };

        double segArea1 = sweepCountPixels(insideSegment, buffer);
        double segArea2 = segmentArea(E, theta0, theta1);

        if (TestParams::debug)
        {
          display(buffer);
          drawTriangle(t, Green8, 3);
        }

        double absError = fabs(segArea1 -segArea2);
        double relError = absError/segArea1;

        const double thres = 0.1;
        EXPECT_NEAR(relError, 0, thres);
        if (TestParams::debug && relError > thres)
        {
          printStage("Numerical error");
          CHECK(i0);
          CHECK(i1);
          CHECK(toDegree(theta0));
          CHECK(toDegree(theta1));
          CHECK(segArea1);
          CHECK(segArea2);
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