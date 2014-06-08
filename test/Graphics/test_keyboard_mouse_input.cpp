#include <gtest/gtest.h>
#include <QtTest>
#include <DO/Graphics.hpp>

using namespace DO;

class TestKeyboardInput: public testing::Test
{
protected:
  Window test_window_;

  TestKeyboardInput()
  {
    test_window_ = openWindow(300, 300);
  }

  virtual ~TestKeyboardInput()
  {
  }
};

TEST_F(TestKeyboardInput, test_getKey)
{
  int expected_key = Qt::Key_A;
  QTest::keyPress(test_window_, Qt::Key_A, Qt::NoModifier, 10);

  int actual_key = getKey();
  
  EXPECT_EQ(actual_key, expected_key);
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
