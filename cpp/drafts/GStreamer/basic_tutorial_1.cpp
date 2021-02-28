#include <gst/gst.h>

int main(int argc, char **argv)
{
  gst_init(&argc, &argv);

  auto pipeline =
      gst_parse_launch("playbin "
                       "uri=https://www.freedesktop.org/software/gstreamer-sdk/"
                       "data/media/sintel_trailer-480p.webm",
                       nullptr);

  // Start processing the stream: display the video stream.
  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  // Keep processing until we reach the end or get an error.
  auto bus = gst_element_get_bus(pipeline);
  auto msg = gst_bus_timed_pop_filtered(
      bus, GST_CLOCK_TIME_NONE,
      GstMessageType(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

  // Cleanup
  if (msg != nullptr)
    gst_message_unref(msg);
  gst_object_unref(bus);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);

  return 0;
}
