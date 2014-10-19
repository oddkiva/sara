TEST(DO_Geometry_Test, coneTest)
{
  Vector2d alpha(1,0), beta(1,1); // generators of the cone
  AffineCone2 convexK(alpha, beta, TestParams::center, AffineCone2::Convex, std::numeric_limits<double>::epsilon());
  AffineCone2 bluntK(alpha, beta, TestParams::center, AffineCone2::Blunt, std::numeric_limits<double>::epsilon());
  AffineCone2 convexPointedK(alpha, alpha, TestParams::center, AffineCone2::Convex);
  AffineCone2 bluntPointedK(alpha, alpha, TestParams::center, AffineCone2::Blunt, std::numeric_limits<double>::epsilon());
  AffineCone2 convexPointedK2(alpha, -alpha, TestParams::center, AffineCone2::Convex, std::numeric_limits<double>::epsilon());
  AffineCone2 bluntPointedK2(alpha, -alpha, TestParams::center, AffineCone2::Blunt, std::numeric_limits<double>::epsilon());

  // ======================================================================== //
  printStage("Convex affine cone testing");
  auto convexPredicate = [&](const Point2d& p) {
    return inside(p, convexK);
  };
  auto convexGroundTruth = [&](const Point2d& p) {
    return
      p.x() > TestParams::w/2. &&
      p.y() > TestParams::h/2. && 
      p.x() > p.y();
  };
  sweepTest(convexPredicate, convexGroundTruth, TestParams::debug);
  if (TestParams::debug)
  {
    drawAffineCone(convexK);
    milliSleep(40);
  }

  // ======================================================================== //
  printStage("Blunt affine cone testing");
  auto bluntPredicate = [&](const Point2d& p) {
    return inside(p, bluntK);
  };
  auto bluntGroundTruth = [&](const Point2d& p) {
    return
      p.x() >= TestParams::w/2. && 
      p.y() >= TestParams::h/2. && 
      p.x() >= p.y();
  };
  sweepTest(bluntPredicate, bluntGroundTruth, TestParams::debug);
  if (TestParams::debug)
  {
    drawAffineCone(bluntK);
    milliSleep(40);
  }

  // ======================================================================== //
  printStage("Convex pointed affine cone testing");
  auto convexPointedPredicate = [&](const Point2d& p) {
    return inside(p, convexPointedK);
  };
  auto convexPointedGroundTruth = [&](const Point2d& p) {
    return false;
  };
  sweepTest(convexPointedPredicate, convexPointedGroundTruth,
    TestParams::debug);
  if (TestParams::debug)
  {
    drawAffineCone(convexPointedK);
    milliSleep(40);
  }

  // ======================================================================== //
  printStage("Blunt pointed affine cone testing");
  auto bluntPointedPredicate = [&](const Point2d& p) {
    return inside(p, bluntPointedK);
  };
  auto bluntPointedGroundTruth = [&](const Point2d& p) {
    return
      p.x() >= TestParams::w/2. && 
      p.y() == TestParams::h/2.;
  };
  sweepTest(bluntPointedPredicate, bluntPointedGroundTruth,
    TestParams::debug);
  if (TestParams::debug)
  {
    drawAffineCone(bluntPointedK);
    milliSleep(40);
  }

  // ======================================================================== //
  printStage("Convex pointed affine cone testing: Case 2");
  auto convexPointedPredicate2 = [&](const Point2d& p) {
    return inside(p, convexPointedK2);
  };
  auto convexPointedGroundTruth2 = [&](const Point2d& p) {
    return false;
  };
  sweepTest(convexPointedPredicate2, convexPointedGroundTruth2,
    TestParams::debug);
  if (TestParams::debug)
  {
    drawAffineCone(convexPointedK2);
    milliSleep(40);
  }

  // ======================================================================== //
  printStage("Blunt pointed affine cone testing: Case 2");
  auto bluntPointedPredicate2 = [&](const Point2d& p) {
    return inside(p, bluntPointedK2);
  };
  auto bluntPointedGroundTruth2 = [&](const Point2d& p) {
    return p.y() == TestParams::h/2.;
  };
  sweepTest(bluntPointedPredicate2, bluntPointedGroundTruth2,
    TestParams::debug);
  if (TestParams::debug)
  {
    drawAffineCone(bluntPointedK2);
    milliSleep(40);
  }

}
