HEADERS += \
    histogrammmatcher.h \
    gabormatcher.h \
    shapematcher.h

SOURCES += \
    histogrammmatcher.cpp \
    gabormatcher.cpp \
    shapematcher.cpp \
    main.cpp

QMAKE_CXXFLAGS += --std=c++11

LIBS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lQt5Core

OTHER_FILES += \
    config.txt \
    ../build-Images-HW4-Desktop-Debug/config.txt
