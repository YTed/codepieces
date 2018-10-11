BIN=kmeans
SRC=kmeans.cpp

all: $(BIN)

$(BIN): $(SRC)
	g++ -o $@ $? `pkg-config --cflags --libs opencv`

clean:
	@rm $(BIN)
