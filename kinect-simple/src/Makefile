APPNAME			= kinect_demo
DESTDIR			= ../bin


#TOOLCHAIN		= /opt/freescale/usr/local/gcc-4.1.2-glibc-2.5-nptl-3/arm-none-linux-gnueabi
#CROSS_COMPILE	= $(TOOLCHAIN)/bin/arm-none-linux-gnueabi-
#CC					= $(CROSS_COMPILE)gcc
#LD					= $(CROSS_COMPILE)g++
CC = gcc
LD = g++

DEL_FILE		= rm -rf
CP_FILE			= cp -rf      


CFLAGS			= -Wall -O2 -fsigned-char -march=armv7-a -mfpu=neon -mfloat-abi=softfp \
                                -I. -I../usr/include \

LFLAGS			= -L../lib -lEGL -lGLESv2 -lpthread \
			-lcxcore -lcv -lcvaux -lhighgui -lml \
			-L/usr/lib \
			-lusb-1.0 -lfreenect \


OBJECTS			= main.o glutils.o glplane.o 


first: all

all: $(APPNAME)

$(APPNAME): $(OBJECTS)
	$(LD) $(LFLAGS) -o $(DESTDIR)/$(APPNAME) $(OBJECTS)

main.o: main.cpp
	$(LD) $(CFLAGS) -c -o main.o main.cpp

glutils.o: glutils.cpp
	$(LD) $(CFLAGS) -c -o glutils.o glutils.cpp

glplane.o: glplane.cpp
	$(LD) $(CFLAGS) -c -o glplane.o glplane.cpp



clean:
	$(DEL_FILE) $(OBJECTS)
	$(DEL_FILE) *~ core *.core

distclean: clean
	$(DEL_FILE) $(DESTDIR)/$(APPNAME)


install: all

