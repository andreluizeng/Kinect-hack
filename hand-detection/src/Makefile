APPNAME			= Hand-detection
DESTDIR			= ../bin

CC = gcc
LD = g++

DEL_FILE		= rm -rf
CP_FILE			= cp -rf      


CFLAGS			= -Wall -O2 -fsigned-char -march=armv7-a -mfpu=neon -mfloat-abi=softfp \
                                -I. -I../usr/include \
				-I/usr/include/gstreamer-0.10 -I/usr/include/glib-2.0 -I/usr/lib/glib-2.0/include -I/usr/include/libxml2 \

LFLAGS			= -L../lib -lEGL -lGLESv2 -lpthread \
			-L/usr/lib -lgstreamer-0.10 -lgobject-2.0 -lgmodule-2.0 -lgthread-2.0 -lrt -lxml2 -lglib-2.0  \
			-lcxcore -lcv -lcvaux -lhighgui -lml \
			-L/usr/lib \
			-lusb-1.0 -lfreenect \


OBJECTS			= main.o glutils.o glcvplane.o gstcontrol.o glcvcube.o


first: all

all: $(APPNAME)

$(APPNAME): $(OBJECTS)
	$(LD) $(LFLAGS) -o $(DESTDIR)/$(APPNAME) $(OBJECTS)

main.o: main.cpp
	$(LD) $(CFLAGS) -c -o main.o main.cpp

glutils.o: glutils.cpp
	$(LD) $(CFLAGS) -c -o glutils.o glutils.cpp

glcvplane.o: glcvplane.cpp
	$(LD) $(CFLAGS) -c -o glcvplane.o glcvplane.cpp

gstcontrol.o: gstcontrol.cpp
	$(LD) $(CFLAGS) -c -o gstcontrol.o gstcontrol.cpp

glcvcube.o: glcvcube.cpp
	$(LD) $(CFLAGS) -c -o glcvcube.o glcvcube.cpp



clean:
	$(DEL_FILE) $(OBJECTS)
	$(DEL_FILE) *~ core *.core

distclean: clean
	$(DEL_FILE) $(DESTDIR)/$(APPNAME)


install: all

