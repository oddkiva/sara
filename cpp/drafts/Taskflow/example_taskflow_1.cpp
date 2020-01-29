#include <thread>

// void VideoOutput::startUp () {
//   _thread = new std::thread ([ this ] () {
//     while (_running) {
//       std::unique_lock<std::mutex> lock (_mutex);
// 
//       _cv.wait (lock, [ this ] () {
//         if (_notified || !_running) {
//           _notified = false;
//           return true;
//         }
// 
//         return _notified;
//       });
// 
//       if (!_running || !_frames.size()) break; // we want to finish the thread
// 
//       _running = _display.show (_frames.front().first, _frames.front().second);
// 
//       _frames.pop();
//     }
//   });
// }
// 
// // ----------------------------------------------------------------------------
// // VideoOutput::display
// // ----------------------------------------------------------------------------
// bool VideoOutput::display (cv::Mat &frame, const FrameData &data) {
//   std::unique_lock<std::mutex> lock (this->_mutex);
// 
//   _frames.emplace (std::make_pair (std::move (frame), std::move (data)));
// 
//   _notified = true;
//   _cv.notify_one();
// 
//   return _running;
// }

auto main() -> int
{
  auto promise = std::promise<int>{};
  auto future = std::future<int>{};
  return 0;
}
