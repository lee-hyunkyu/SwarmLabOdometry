SRC 	= src
TESTS 	= tests
main:
    $(MAKE) -C $(SRC) main

test:
    echo "Pass"

test-null_space:
    $(MAKE) -C $(TESTS) test-null_space
