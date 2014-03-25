TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    knnfinder.cpp

LIBS += -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_flann -lopencv_nonfree -lopencv_features2d -lopencv_imgproc -lQtCore

INCLUDEPATH += /usr/include/qt4 /usr/include/qt4/QtCore

QMAKE_CXXFLAGS += -std=c++11

HEADERS += \
    knnfinder.h \
    clusterspace.h \
    Cluster.h

