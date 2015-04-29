#include <cstdarg>
#include <cstdio>
#include <vector>

#include <DO/Core/StringFormat.hpp>


namespace DO {

   static std::string vformat(const char *format, va_list args)
   {
      size_t size = 1024;
      std::vector<char> buffer(size);

      while (true)
      {
         int formatted_string_length = vsnprintf(buffer.data(), size, format, args);

         if (formatted_string_length <= int(size) && formatted_string_length >= 0)
            return std::string(buffer.data(), size_t(formatted_string_length));

         size = formatted_string_length > 0 ? formatted_string_length+1 : size*2;
         buffer.resize(size);
      }
   }

   std::string format(const char *fmt, ...)
   {
      va_list args;
      va_start(args, fmt);
      std::string formatted_message = vformat(fmt, args);
      va_end(args);
      return formatted_message;
   }

}
