import os, shutil, re

msvc_dir = 'msvc10'
output_dir = [ 'lib', 'bin' ]
build_type = [ 'Debug', 'Release' ]
lib_type = [ '', 'SHARED' ]

packaged_lib_dir = 'packaged-libs'
if not os.path.isdir(packaged_lib_dir):
    os.mkdir(packaged_lib_dir)

packaged_lib_msvc_dir = packaged_lib_dir+'/'+msvc_dir
if not os.path.isdir(packaged_lib_msvc_dir):
    os.mkdir(packaged_lib_msvc_dir)

for lib_type_dir in output_dir:

    target_dir = packaged_lib_dir+'/'+msvc_dir+'/'+lib_type_dir
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
            
    for config_type_dir in build_type:
        source_dir = lib_type_dir+'/'+msvc_dir+'/'+config_type_dir

        for file in os.listdir(source_dir):
            shutil.copy(source_dir+'/'+file, target_dir+'/'+file)
