OBJS = main.o perceptron.o
CC = g++
DEBUG = -g -O3
CFLAGS = -std=c++11 -Wall -fPIC -c $(DEBUG)
LFLAGS = -Wall $(DEBUG)

runProg : $(OBJS)
	$(CC) $(LFLAGS) $(OBJS) -o runProg

main.o : main.cpp perceptron.cpp perceptron.hpp
	$(CC) $(CFLAGS) main.cpp

perceptron.o : perceptron.cpp perceptron.hpp
	$(CC) $(CFLAGS) perceptron.cpp

clean:	
	\rm *.o runProg