CC=g++

SOURCES=main.cpp BRIEF.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=BRIEF_demo

CFLAGS = -Wall -DHAVE_INLINE -msse4.2 -O3 # -g

# If your processor does not support SSE 4.2 instructions
# comment the line above with a '#' and uncomment the line below

#CFLAGS = -Wall -DHAVE_INLINE -O3 # -g

# OpenCV
CFLAGS += `pkg-config opencv --cflags`
LDFLAGS += `pkg-config opencv --libs`

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -f *.o $(EXECUTABLE) *~
