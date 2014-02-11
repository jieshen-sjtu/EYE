import os
import sys

SRC = Split('''
            src/eye_llc.cpp
            src/eye_codebook.cpp
            src/eye_dsift.cpp
            src/test.cpp
            src/main.cpp
            ''')


INCLUDE_PATH = Split('''./include
                        /usr/include
                        /usr/local/include/
                        /opt/intel/mkl/include/
                        ''')

LIB_PATH = Split('''./lib
                    /usr/local/lib
                    /usr/lib
                    /opt/intel/mkl/lib/intel64/
                    ''')

_LIBS = Split('''
                 libvl.so
                 libmkl_rt.so
                ''')


env = Environment(LIBPATH=LIB_PATH, LIBS=_LIBS, CPPPATH=INCLUDE_PATH, LINKFLAGS='-fopenmp',
                  CFLAGS='-O3', CXXFLAGS='-O3', CXX='g++');
env.ParseConfig('pkg-config --cflags --libs opencv');

env.StaticLibrary(target='EYE', source=SRC)
env.Program(target='EYE.bin', source=SRC)
