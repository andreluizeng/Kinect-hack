APPNAME			= Hand-detection
DESTDIR			= ../bin

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


OBJECTS			= main.o glutils.o glcvplane.o 


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



clean:
	$(DEL_FILE) $(OBJECTS)
	$(DEL_FILE) *~ core *.core

distclean: clean
	$(DEL_FILE) $(DESTDIR)/$(APPNAME)


install: all
