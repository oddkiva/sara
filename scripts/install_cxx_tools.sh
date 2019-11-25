#!/bin/bash
set -ex

sudo apt install ccache
sudo touch /usr/local/bin/ccache-clang
echo "#!/bin/bash" >> /usr/local/bin/ccache-clang
echo "exec ccache /usr/bin/clang-6.0 "\$\@"" >> /usr/local/bin/ccache-clang

sudo touch /usr/local/bin/ccache-clang++
echo "#!/bin/bash" >> /usr/local/bin/ccache-clang
echo "exec ccache /usr/bin/clang++-6.0 "\$\@"" >> /usr/local/bin/ccache-clang
