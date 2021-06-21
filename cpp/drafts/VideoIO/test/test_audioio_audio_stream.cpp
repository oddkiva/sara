#define BOOST_TEST_MODULE "AudioIO/AudioStream Class"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core.hpp>
#include <DO/Sara/VideoIO/AudioStream.hpp>


using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestAudioStream)

BOOST_AUTO_TEST_CASE(test)
{
  test_audio("/Users/david/Downloads/bensound-tenderness.mp3",
             "/Users/david/Desktop/out.mp3");
}

BOOST_AUTO_TEST_SUITE_END()
