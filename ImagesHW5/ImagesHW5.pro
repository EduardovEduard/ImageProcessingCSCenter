TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    siftmatcher.cpp \
    harrismatcher.cpp

QMAKE_CXXFLAGS += --std=c++11

LIBS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_nonfree \
        -lopencv_objdetect -lopencv_features2d -lopencv_contrib\
        -lopencv_flann -lopencv_stitching

HEADERS += \
    blobdetector.hpp \
    siftmatcher.hpp \
    harrismatcher.hpp \
    constants.h
