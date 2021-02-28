#include <gst/gst.h>

int main(int argc, char** argv)
{
  gst_init(&argc, &argv);

  auto source = gst_element_factory_make("videotestsrc", "source");
  auto sink = gst_element_factory_make("autovideosink", "sink");

  auto pipeline = gst_pipeline_new("test-pipeline");
  if (!pipeline || !source || !sink)
  {
    g_printerr("Not all elements could be created!\n");
    return -1;
  }

  gst_bin_add_many(GST_BIN(pipeline), source, sink, nullptr);
  if (gst_element_link(source, sink) != TRUE)
  {
    g_printerr("Elements could not be linked!\n");
    gst_object_unref(pipeline);
    return -1;
  }

  g_object_set(source, "pattern", 0, nullptr);
  auto ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE)
  {
    g_printerr("Unable to set the pipeline to the playing state!\n");
    gst_object_unref(pipeline);
    return -1;
  }

  auto bus = gst_element_get_bus(pipeline);
  auto msg = gst_bus_timed_pop_filtered(
      bus, GST_CLOCK_TIME_NONE,
      GstMessageType(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

  if (msg != nullptr)
  {
    GError* err = nullptr;
    gchar* debug_info = nullptr;

    switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_ERROR:
      gst_message_parse_error(msg, &err, &debug_info);
      g_printerr("Error received from element %s: %s\n",
                 GST_OBJECT_NAME(msg->src), err->message);
      g_printerr("Debugging information: %s\n",
                 debug_info ? debug_info : "none");
      g_clear_error(&err);
      g_free(debug_info);
      break;
    case GST_MESSAGE_EOS:
      g_print("End of stream reached!\n");
      break;
    default:
      g_printerr("Unexpected message received!\n");
      break;
    }

    gst_message_unref(msg);
  }

  gst_object_unref(bus);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);

  return 0;
}
