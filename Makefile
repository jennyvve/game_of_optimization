# tested with: g++-11
CC 		= gcc-14
CFLAGS  = -mavx2 -O5
SRC 	= src
TARGET 	= life.bin

life.bin: src/main.c
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)/main.c

all: life.bin

clean:
	$(RM) $(TARGET)